"""
Ablation B: StructTimeVAE with STATIC scene encoding.

Instead of the full dynamic scene encoder (scene_rnn over T observed frames +
scene_time_gru extrapolation), this ablation uses a single static scene embedding:
  - Mean-pool DINOv2 embeddings over T observation frames -> [B, dino_embed_dim]
  - Project to hidden_dim via scene_proj
  - Repeat the same vector for all prediction timesteps in the decoder

This isolates the contribution of temporal scene dynamics:
  Full model ADE - Ablation B ADE = gain from dynamic scene temporal evolution
  Ablation B ADE - Ablation A ADE = gain from static scene context alone

Prior and posterior are conditioned on [h_agent, h_social, h_scene_static]
identical to the full model's use of h_scene_summary, but h_scene_static is
just the mean-pooled projection rather than the GRU final hidden state.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class StructTimeVAE_StaticScene(nn.Module):

    PADDING_SENTINEL = 1e8

    class SocialAttention(nn.Module):
        def __init__(self, hidden_dim: int, neighbor_feat_dim: int = 5):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.key_embed  = nn.Sequential(nn.Linear(neighbor_feat_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            self.val_embed  = nn.Sequential(nn.Linear(neighbor_feat_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            self.query_proj = nn.Linear(hidden_dim, hidden_dim)
            self.out_proj   = nn.Linear(hidden_dim, hidden_dim)

        def forward(self, h_agent, x, neighbor):
            x_last = x[-1]; n_last = neighbor[-1]
            B, N, F = n_last.shape
            pad_mask  = n_last[:, :, 0] < StructTimeVAE_StaticScene.PADDING_SENTINEL
            n_clamped = n_last * pad_mask.unsqueeze(-1).float()
            rel_pos   = n_clamped[:, :, :2]  - x_last[:, :2].unsqueeze(1)
            rel_vel   = n_clamped[:, :, 2:4] - x_last[:, 2:4].unsqueeze(1)
            distance  = rel_pos.norm(dim=-1, keepdim=True)
            rel_feat  = torch.cat([rel_pos, rel_vel, distance], dim=-1) * pad_mask.unsqueeze(-1).float()
            keys      = self.key_embed(rel_feat)
            values    = self.val_embed(rel_feat)
            query     = self.query_proj(h_agent).unsqueeze(2)
            scores    = torch.bmm(keys, query).squeeze(2) / (self.hidden_dim ** 0.5)
            pad_scores  = scores.masked_fill(~pad_mask, float('-inf'))
            all_padded  = ~pad_mask.any(dim=-1)
            safe_scores = pad_scores.masked_fill(all_padded.unsqueeze(1).expand_as(pad_scores), 0.0)
            attn = torch.softmax(safe_scores, dim=-1).masked_fill(all_padded.unsqueeze(1), 0.0)
            return self.out_proj(torch.bmm(attn.unsqueeze(1), values).squeeze(1))

    class LatentNet(nn.Module):
        def __init__(self, input_dim, z_dim):
            super().__init__()
            self.net    = nn.Sequential(nn.Linear(input_dim, input_dim), nn.ReLU(), nn.Linear(input_dim, input_dim), nn.ReLU())
            self.mu     = nn.Linear(input_dim, z_dim)
            self.logvar = nn.Linear(input_dim, z_dim)
        def forward(self, x):
            h = self.net(x); return self.mu(h), self.logvar(h).clamp(-4, 4)

    class ParallelDecoder(nn.Module):
        def __init__(self, hidden_dim, z_agent_dim, scene_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(hidden_dim + z_agent_dim + scene_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, 2),
            )
        def forward(self, h_agent, z_agent, c_scene_seq):
            T = c_scene_seq.size(1)
            dec_in = torch.cat([h_agent.unsqueeze(1).expand(-1,T,-1), z_agent.unsqueeze(1).expand(-1,T,-1), c_scene_seq], dim=-1)
            return self.net(dec_in)

    def __init__(self, horizon, input_dim=6, hidden_dim=256, z_agent_dim=32,
                 scene_dim=32, dino_embed_dim=768, kl_beta_agent=1.0,
                 kl_beta_scene=1.0, free_bits=0.01, dinov2_path=None, device="cuda:0"):
        super().__init__()
        self.horizon       = horizon
        self.input_dim     = input_dim
        self.hidden_dim    = hidden_dim
        self.z_agent_dim   = z_agent_dim
        self.scene_dim     = scene_dim
        self.kl_beta_agent = kl_beta_agent
        self.free_bits     = free_bits
        self.device        = torch.device(device)

        self.agent_embed  = nn.Linear(input_dim, hidden_dim)
        self.agent_rnn    = nn.GRU(hidden_dim, hidden_dim)
        self.social_attn  = self.SocialAttention(hidden_dim)
        self.future_embed = nn.Linear(2, hidden_dim)
        self.future_rnn   = nn.GRU(hidden_dim, hidden_dim)

        # Static scene encoder: project mean-pooled DINOv2 embedding
        self.scene_proj = nn.Linear(dino_embed_dim, hidden_dim)
        self.scene_norm = nn.LayerNorm(hidden_dim)

        # Project hidden_dim -> scene_dim for decoder input
        self.scene_to_dec = nn.Sequential(
            nn.Linear(hidden_dim, scene_dim),
            nn.ReLU(),
        )

        # Prior/posterior: [h_agent, h_social, h_scene_static] = 3H
        prior_ctx_dim = 3 * hidden_dim
        post_ctx_dim  = 4 * hidden_dim
        self.p_z_agent = self.LatentNet(prior_ctx_dim, z_agent_dim)
        self.q_z_agent = self.LatentNet(post_ctx_dim,  z_agent_dim)
        self.decoder   = self.ParallelDecoder(hidden_dim, z_agent_dim, scene_dim)
        self.to(self.device)

    @staticmethod
    def _reparameterize(mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    @staticmethod
    def _kl_gaussian(mu_q, logvar_q, mu_p, logvar_p, free_bits=0.01):
        kl = 0.5 * (logvar_p - logvar_q + (logvar_q.exp() + (mu_q-mu_p).pow(2)) / (logvar_p.exp()+1e-8) - 1.0)
        return torch.clamp(kl, min=free_bits).sum(dim=-1).mean()

    def encode_agent(self, x):
        x = x.to(self.device, non_blocking=True)
        L, B, F = x.shape
        x_emb = self.agent_embed(x.reshape(L*B, F)).view(L, B, self.hidden_dim)
        _, h  = self.agent_rnn(x_emb)
        return h.squeeze(0)

    def encode_social(self, h_agent, x, neighbor, last_obs):
        if neighbor is None:
            return torch.zeros(h_agent.size(0), self.hidden_dim, device=self.device)
        neighbor = neighbor.to(self.device, non_blocking=True)
        x_dev    = x.to(self.device, non_blocking=True)
        if last_obs is not None:
            last_obs    = last_obs.to(self.device, non_blocking=True)
            pad_mask_4  = neighbor[:, :, :, 0] < self.PADDING_SENTINEL
            shifted_pos = neighbor[:, :, :, :2] - last_obs.unsqueeze(0).unsqueeze(2)
            neighbor    = torch.cat([shifted_pos, neighbor[:, :, :, 2:]], dim=-1)
            sentinel    = torch.full_like(neighbor, self.PADDING_SENTINEL)
            keep        = pad_mask_4.unsqueeze(-1).expand_as(neighbor)
            neighbor    = torch.where(keep, neighbor, sentinel)
        return self.social_attn(h_agent, x_dev, neighbor)

    def encode_scene_static(self, scene_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        scene_embeddings: [B, T, dino_embed_dim]
        Returns:
          h_scene_static : [B, hidden_dim]   for prior/posterior context
          c_scene_seq    : [B, horizon, scene_dim]  same vector repeated for decoder
        """
        scene_embeddings = scene_embeddings.to(self.device, dtype=torch.float32, non_blocking=True)
        # Mean-pool over observation timesteps
        pooled = scene_embeddings.mean(dim=1)                    # [B, dino_embed_dim]
        h_scene_static = self.scene_norm(self.scene_proj(pooled))  # [B, hidden_dim]
        # Project to scene_dim and repeat for all prediction timesteps
        c_static = self.scene_to_dec(h_scene_static)              # [B, scene_dim]
        c_scene_seq = c_static.unsqueeze(1).expand(-1, self.horizon, -1)  # [B, horizon, scene_dim]
        return h_scene_static, c_scene_seq

    def encode_future(self, y):
        y = y.to(self.device, non_blocking=True)
        T, B, D = y.shape
        y_emb = self.future_embed(y.reshape(T*B, D)).view(T, B, self.hidden_dim)
        _, h  = self.future_rnn(y_emb)
        return h.squeeze(0)

    def forward(self, x, scene_embeddings=None, y=None,
                neighbor=None, last_obs=None,
                kl_beta_agent=None, kl_beta_scene=None):
        x        = x.to(self.device, non_blocking=True)
        h_agent  = self.encode_agent(x)
        B        = h_agent.size(0)
        h_social = self.encode_social(h_agent, x, neighbor, last_obs)

        if scene_embeddings is not None:
            h_scene_static, c_scene_seq = self.encode_scene_static(scene_embeddings)
        else:
            h_scene_static = torch.zeros(B, self.hidden_dim, device=self.device)
            c_scene_seq    = torch.zeros(B, self.horizon, self.scene_dim, device=self.device)

        # Prior: [h_agent, h_social, h_scene_static]
        prior_ctx          = torch.cat([h_agent, h_social, h_scene_static], dim=-1)
        mu_p_a, logvar_p_a = self.p_z_agent(prior_ctx)

        if y is None:
            z_agent = self._reparameterize(mu_p_a, logvar_p_a)
            return self.decoder(h_agent, z_agent, c_scene_seq)

        y = y.to(self.device, non_blocking=True)
        if y.dim() == 4:
            y = y.squeeze(0) if y.size(0) == 1 else y.squeeze(1)
        if y.shape != (B, self.horizon, 2):
            raise ValueError(f"Expected y [{B},{self.horizon},2], got {y.shape}")

        h_future           = self.encode_future(y.permute(1, 0, 2).contiguous())
        post_ctx           = torch.cat([h_agent, h_social, h_scene_static, h_future], dim=-1)
        mu_q_a, logvar_q_a = self.q_z_agent(post_ctx)
        z_agent            = self._reparameterize(mu_q_a, logvar_q_a)
        pred               = self.decoder(h_agent, z_agent, c_scene_seq)
        rec                = F.mse_loss(pred, y)
        kl_agent           = self._kl_gaussian(mu_q_a, logvar_q_a, mu_p_a, logvar_p_a, self.free_bits)
        beta_a             = self.kl_beta_agent if kl_beta_agent is None else kl_beta_agent
        return {
            "loss":     rec + beta_a * kl_agent,
            "rec":      rec,
            "kl_agent": kl_agent,
            "kl_scene": torch.tensor(0.0, device=self.device),
            "pred":     pred,
        }

    @torch.no_grad()
    def sample(self, x, scene_embeddings=None, n_samples=20, neighbor=None, last_obs=None):
        was_training = self.training; self.eval()
        try:
            return torch.stack([
                self.forward(x, scene_embeddings, None, neighbor, last_obs)
                for _ in range(n_samples)
            ])
        finally:
            self.train(was_training)