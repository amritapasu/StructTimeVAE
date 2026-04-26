from typing import Optional, List, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class StructTimeVAE(nn.Module):
    """
    StructTimeVAE — merged architecture:

    Scene stream (from partner):
      - Pre-computed DINOv2 embeddings [B, T, dino_embed_dim] from cache
        (one embedding per observation timestep — dynamic scene encoding)
      - scene_proj + scene_rnn encodes T observed frames -> h_scene [B, T, hidden_dim]
      - scene_time_gru extrapolates from observed scene dynamics into pred horizon
      - Scene context is DETERMINISTIC (no q/p split for scene — avoids collapse)

    Agent stream (from our version):
      - GRU over observed displacement sequence -> h_agent [B, hidden_dim]
      - Stochastic CVAE latent z_agent with correct prior/posterior split:
          prior   p(z_agent | h_agent, h_social, h_scene_summary)
          posterior q(z_agent | h_agent, h_social, h_scene_summary, h_future)
      - Free bits KL floor prevents posterior collapse

    Social stream (from our version):
      - Masked dot-product attention over neighbor relative states
      - Padding sentinel 1e9 detected and masked before softmax
      - Agents with no neighbors receive zero social context

    Input conventions:
      x                : [obs_len, B, input_dim]       displacement-space agent history
      scene_embeddings : [B, T, dino_embed_dim]        pre-computed per-frame DINOv2 embeddings
      neighbor         : [obs_len, B, N_max, 6]        neighbor states (absolute coords)
      last_obs         : [B, 2]                        last observed absolute position
      y                : [B, pred_len, 2]              future displacements (training only)
    """

    PADDING_SENTINEL = 1e8

    # --------------------------------------------------
    # SocialAttention
    # --------------------------------------------------

    class SocialAttention(nn.Module):
        """
        Masked scaled dot-product attention over neighbor relative states.
        Uses last observation timestep only.
        Relative features: [rel_pos(2), rel_vel(2), distance(1)] = 5 dims
        """
        def __init__(self, hidden_dim: int, neighbor_feat_dim: int = 5):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.key_embed = nn.Sequential(
                nn.Linear(neighbor_feat_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.val_embed = nn.Sequential(
                nn.Linear(neighbor_feat_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.query_proj = nn.Linear(hidden_dim, hidden_dim)
            self.out_proj   = nn.Linear(hidden_dim, hidden_dim)

        def forward(
            self,
            h_agent:  torch.Tensor,   # [B, hidden_dim]
            x:        torch.Tensor,   # [obs_len, B, 6]
            neighbor: torch.Tensor,   # [obs_len, B, N, 6]
        ) -> torch.Tensor:            # [B, hidden_dim]
            x_last = x[-1]            # [B, 6]
            n_last = neighbor[-1]     # [B, N, 6]
            B, N, F = n_last.shape

            # Padding mask: True = real neighbor
            pad_mask = n_last[:, :, 0] < StructTimeVAE.PADDING_SENTINEL  # [B, N]

            # Zero padded entries before computing relative features
            n_clamped = n_last * pad_mask.unsqueeze(-1).float()

            target_pos = x_last[:, :2].unsqueeze(1)    # [B, 1, 2]
            target_vel = x_last[:, 2:4].unsqueeze(1)   # [B, 1, 2]

            rel_pos  = n_clamped[:, :, :2]  - target_pos   # [B, N, 2]
            rel_vel  = n_clamped[:, :, 2:4] - target_vel   # [B, N, 2]
            distance = rel_pos.norm(dim=-1, keepdim=True)   # [B, N, 1]

            rel_feat = torch.cat([rel_pos, rel_vel, distance], dim=-1)  # [B, N, 5]
            rel_feat = rel_feat * pad_mask.unsqueeze(-1).float()

            keys   = self.key_embed(rel_feat)    # [B, N, hidden_dim]
            values = self.val_embed(rel_feat)    # [B, N, hidden_dim]
            query  = self.query_proj(h_agent).unsqueeze(2)  # [B, hidden_dim, 1]

            scale  = self.hidden_dim ** 0.5
            scores = torch.bmm(keys, query).squeeze(2) / scale  # [B, N]

            # Mask padding out-of-place
            pad_scores  = scores.masked_fill(~pad_mask, float('-inf'))
            all_padded  = ~pad_mask.any(dim=-1)
            safe_scores = pad_scores.masked_fill(
                all_padded.unsqueeze(1).expand_as(pad_scores), 0.0
            )

            attn = torch.softmax(safe_scores, dim=-1)
            attn = attn.masked_fill(all_padded.unsqueeze(1), 0.0)

            h_social = torch.bmm(attn.unsqueeze(1), values).squeeze(1)
            return self.out_proj(h_social)

    # --------------------------------------------------
    # LatentNet
    # --------------------------------------------------

    class LatentNet(nn.Module):
        def __init__(self, input_dim: int, z_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.ReLU(),
                nn.Linear(input_dim, input_dim),
                nn.ReLU(),
            )
            self.mu     = nn.Linear(input_dim, z_dim)
            self.logvar = nn.Linear(input_dim, z_dim)

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            h      = self.net(x)
            mu     = self.mu(h)
            logvar = self.logvar(h).clamp(-4.0, 4.0)
            return mu, logvar

    # --------------------------------------------------
    # ParallelDecoder
    # --------------------------------------------------

    class ParallelDecoder(nn.Module):
        def __init__(self, hidden_dim: int, z_agent_dim: int, scene_dim: int):
            super().__init__()
            in_dim = hidden_dim + z_agent_dim + scene_dim
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2),
            )

        def forward(
            self,
            h_agent:     torch.Tensor,   # [B, hidden_dim]
            z_agent:     torch.Tensor,   # [B, z_agent_dim]
            c_scene_seq: torch.Tensor,   # [B, horizon, scene_dim]
        ) -> torch.Tensor:
            horizon     = c_scene_seq.size(1)
            h_agent_exp = h_agent.unsqueeze(1).expand(-1, horizon, -1)
            z_agent_exp = z_agent.unsqueeze(1).expand(-1, horizon, -1)
            dec_in      = torch.cat([h_agent_exp, z_agent_exp, c_scene_seq], dim=-1)
            return self.net(dec_in)   # [B, horizon, 2]

    # --------------------------------------------------
    # __init__
    # --------------------------------------------------

    def __init__(
        self,
        horizon:        int,
        input_dim:      int   = 6,
        hidden_dim:     int   = 256,
        z_agent_dim:    int   = 32,
        scene_dim:      int   = 32,
        dino_embed_dim: int   = 768,    # ViT-B/14 output dim from cache
        kl_beta_agent:  float = 1.0,
        kl_beta_scene:  float = 1.0,    # kept for API compatibility, unused
        free_bits:      float = 0.01,
        dinov2_path:    Optional[str] = None,   # unused, kept for API compat
        device:         str   = "cuda:0",
    ):
        super().__init__()

        self.horizon        = horizon
        self.input_dim      = input_dim
        self.hidden_dim     = hidden_dim
        self.z_agent_dim    = z_agent_dim
        self.scene_dim      = scene_dim
        self.kl_beta_agent  = kl_beta_agent
        self.free_bits      = free_bits
        self.device         = torch.device(device)

        # --------------------------
        # Agent history encoder
        # --------------------------
        self.agent_embed = nn.Linear(input_dim, hidden_dim)
        self.agent_rnn   = nn.GRU(hidden_dim, hidden_dim)

        # --------------------------
        # Social attention
        # --------------------------
        self.social_attn = self.SocialAttention(hidden_dim, neighbor_feat_dim=5)

        # --------------------------
        # Future encoder (posterior only, training only)
        # --------------------------
        self.future_embed = nn.Linear(2, hidden_dim)
        self.future_rnn   = nn.GRU(hidden_dim, hidden_dim)

        # --------------------------
        # Scene encoder (dynamic, from partner)
        #
        # Receives pre-computed DINOv2 embeddings [B, T, dino_embed_dim]
        # where T = obs_len. One embedding per observation frame.
        # scene_proj:  project 768 -> hidden_dim per frame
        # scene_rnn:   GRU over T frames -> h_scene [B, T, hidden_dim]
        #              captures how the scene changes during observation
        # --------------------------
        self.scene_proj = nn.Linear(dino_embed_dim, hidden_dim)
        self.scene_rnn  = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        # Scene summary for prior/posterior context:
        # project final scene hidden state to hidden_dim
        self.scene_summary_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # --------------------------
        # Scene temporal GRU (from partner, improved)
        #
        # Two-pass design:
        #   Pass 1: run scene_time_gru over observed scene latents
        #           (T steps) to build hidden state h0 capturing
        #           observed scene dynamics
        #   Pass 2: free-unroll from h0 for horizon steps with zero
        #           inputs -- extrapolates scene dynamics into future
        #
        # scene_h0_proj: projects h_scene summary to z_scene_dim
        #                to seed the scene_time_gru hidden state
        # --------------------------
        self.scene_h0_proj  = nn.Sequential(
            nn.Linear(hidden_dim, scene_dim),
            nn.Tanh(),
        )
        self.scene_time_gru = nn.GRU(scene_dim, scene_dim, batch_first=True)

        # --------------------------
        # z_agent latent networks
        #
        # prior_ctx:  [h_agent, h_social, h_scene_summary]        = 3H
        # post_ctx:   [h_agent, h_social, h_scene_summary, h_fut] = 4H
        # --------------------------
        prior_ctx_dim = 3 * hidden_dim
        post_ctx_dim  = 4 * hidden_dim

        self.p_z_agent = self.LatentNet(prior_ctx_dim, z_agent_dim)
        self.q_z_agent = self.LatentNet(post_ctx_dim,  z_agent_dim)

        # --------------------------
        # Decoder
        # --------------------------
        self.decoder = self.ParallelDecoder(hidden_dim, z_agent_dim, scene_dim)

        self.to(self.device)

    # --------------------------------------------------
    # Static utilities
    # --------------------------------------------------

    @staticmethod
    def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    @staticmethod
    def _kl_gaussian(
        mu_q:      torch.Tensor,
        logvar_q:  torch.Tensor,
        mu_p:      torch.Tensor,
        logvar_p:  torch.Tensor,
        free_bits: float = 0.01,
    ) -> torch.Tensor:
        var_q = logvar_q.exp()
        var_p = logvar_p.exp()
        kl_per_dim = 0.5 * (
            logvar_p - logvar_q
            + (var_q + (mu_q - mu_p).pow(2)) / (var_p + 1e-8)
            - 1.0
        )
        return torch.clamp(kl_per_dim, min=free_bits).sum(dim=-1).mean()

    # --------------------------------------------------
    # Encoders
    # --------------------------------------------------

    def encode_agent(self, x: torch.Tensor) -> torch.Tensor:
        """x: [obs_len, B, input_dim] -> [B, hidden_dim]"""
        x = x.to(self.device, non_blocking=True)
        L, B, F = x.shape
        if F != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {F}")
        x_emb = self.agent_embed(x.reshape(L * B, F)).view(L, B, self.hidden_dim)
        _, h  = self.agent_rnn(x_emb)
        return h.squeeze(0)

    def encode_social(
        self,
        h_agent:  torch.Tensor,
        x:        torch.Tensor,
        neighbor: Optional[torch.Tensor],
        last_obs: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Returns h_social [B, hidden_dim]. Zeros if no neighbors."""
        if neighbor is None:
            return torch.zeros(h_agent.size(0), self.hidden_dim, device=self.device)

        neighbor = neighbor.to(self.device, non_blocking=True)
        x_dev    = x.to(self.device, non_blocking=True)

        if last_obs is not None:
            last_obs = last_obs.to(self.device, non_blocking=True)
            pad_mask_4 = (neighbor[:, :, :, 0] < self.PADDING_SENTINEL)
            shifted_pos = neighbor[:, :, :, :2] - last_obs.unsqueeze(0).unsqueeze(2)
            neighbor    = torch.cat([shifted_pos, neighbor[:, :, :, 2:]], dim=-1)
            sentinel    = torch.full_like(neighbor, self.PADDING_SENTINEL)
            keep        = pad_mask_4.unsqueeze(-1).expand_as(neighbor)
            neighbor    = torch.where(keep, neighbor, sentinel)

        return self.social_attn(h_agent, x_dev, neighbor)

    def encode_scene(
        self,
        scene_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        scene_embeddings: [B, T, dino_embed_dim]  pre-computed per-frame embeddings

        Returns:
          h_scene        : [B, T, hidden_dim]  per-frame hidden states
          h_scene_summary: [B, hidden_dim]     final hidden state summary
                                               (used in prior/posterior context)
        """
        scene_embeddings = scene_embeddings.to(
            dtype=torch.float32, device=self.device, non_blocking=True
        )
        projected = self.scene_proj(scene_embeddings)   # [B, T, hidden_dim]
        h_scene, h_final = self.scene_rnn(projected)    # h_scene: [B, T, hidden_dim]
                                                        # h_final: [1, B, hidden_dim]
        h_scene_summary = self.scene_summary_proj(
            h_final.squeeze(0)                          # [B, hidden_dim]
        )
        return h_scene, h_scene_summary

    def encode_future(self, y: torch.Tensor) -> torch.Tensor:
        """y: [pred_len, B, 2] time-first -> [B, hidden_dim]. Training only."""
        y = y.to(self.device, non_blocking=True)
        T, B, D = y.shape
        y_emb = self.future_embed(y.reshape(T * B, D)).view(T, B, self.hidden_dim)
        _, h  = self.future_rnn(y_emb)
        return h.squeeze(0)

    # --------------------------------------------------
    # Scene temporal evolution (two-pass, from partner)
    # --------------------------------------------------

    def evolve_scene_context(
        self,
        h_scene: torch.Tensor,   # [B, T, hidden_dim]
    ) -> torch.Tensor:            # [B, horizon, scene_dim]
        """
        Two-pass scene temporal evolution:

        Pass 1 (observe): project h_scene to scene_dim, run scene_time_gru
                          over T observed frames to build hidden state h0
                          that captures observed scene dynamics.

        Pass 2 (extrapolate): free-unroll scene_time_gru for horizon steps
                              from h0 using zero inputs. This extrapolates
                              the learned scene dynamics into the prediction
                              horizon without any new visual input.

        This is the key dynamic scene contribution: unlike the static version
        which evolved a single embedding, this version grounds the temporal
        evolution in T actually-observed video frames.
        """
        B, T, _ = h_scene.shape

        # Project observed scene hidden states to scene_dim
        c_obs = self.scene_h0_proj(
            h_scene.reshape(B * T, self.hidden_dim)
        ).reshape(B, T, self.scene_dim)                  # [B, T, scene_dim]

        # Pass 1: encode observed scene sequence
        _, h0 = self.scene_time_gru(c_obs)               # h0: [1, B, scene_dim]

        # Pass 2: extrapolate into prediction horizon
        zeros = torch.zeros(B, self.horizon, self.scene_dim, device=h_scene.device)
        c_scene_seq, _ = self.scene_time_gru(zeros, h0)  # [B, horizon, scene_dim]

        return c_scene_seq

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------

    def forward(
        self,
        x:                torch.Tensor,
        scene_embeddings: Optional[torch.Tensor] = None,
        y:                Optional[torch.Tensor] = None,
        neighbor:         Optional[torch.Tensor] = None,
        last_obs:         Optional[torch.Tensor] = None,
        kl_beta_agent:    Optional[float] = None,
        kl_beta_scene:    Optional[float] = None,   # accepted, unused
    ):
        """
        Inference (y=None):  returns pred [B, T, 2]
        Training  (y given): returns dict: loss, rec, kl_agent, kl_scene, pred
        """
        x       = x.to(self.device, non_blocking=True)
        h_agent = self.encode_agent(x)
        B       = h_agent.size(0)

        # Social context
        h_social = self.encode_social(h_agent, x, neighbor, last_obs)  # [B, hidden_dim]

        # Scene context (dynamic per-frame)
        if scene_embeddings is not None:
            h_scene, h_scene_summary = self.encode_scene(scene_embeddings)
            c_scene_seq = self.evolve_scene_context(h_scene)           # [B, horizon, scene_dim]
        else:
            h_scene_summary = torch.zeros(B, self.hidden_dim, device=self.device)
            c_scene_seq     = torch.zeros(B, self.horizon, self.scene_dim, device=self.device)

        # Prior: conditioned on agent + social + scene summary
        prior_ctx          = torch.cat([h_agent, h_social, h_scene_summary], dim=-1)  # [B, 3H]
        mu_p_a, logvar_p_a = self.p_z_agent(prior_ctx)

        if y is None:
            z_agent = self._reparameterize(mu_p_a, logvar_p_a)
            return self.decoder(h_agent, z_agent, c_scene_seq)

        # Training: posterior
        y = y.to(self.device, non_blocking=True)
        if y.dim() == 4:
            if y.size(0) == 1:
                y = y.squeeze(0)
            elif y.size(1) == 1:
                y = y.squeeze(1)
        if y.shape != (B, self.horizon, 2):
            raise ValueError(
                f"Expected y [B={B}, horizon={self.horizon}, 2], got {y.shape}."
            )

        y_tb     = y.permute(1, 0, 2).contiguous()
        h_future = self.encode_future(y_tb)

        post_ctx           = torch.cat([h_agent, h_social, h_scene_summary, h_future], dim=-1)  # [B, 4H]
        mu_q_a, logvar_q_a = self.q_z_agent(post_ctx)

        z_agent  = self._reparameterize(mu_q_a, logvar_q_a)
        pred     = self.decoder(h_agent, z_agent, c_scene_seq)

        rec      = F.mse_loss(pred, y)
        kl_agent = self._kl_gaussian(
            mu_q_a, logvar_q_a, mu_p_a, logvar_p_a, self.free_bits
        )
        kl_scene = torch.tensor(0.0, device=self.device)

        beta_a = self.kl_beta_agent if kl_beta_agent is None else kl_beta_agent
        loss   = rec + beta_a * kl_agent

        return {
            "loss":     loss,
            "rec":      rec,
            "kl_agent": kl_agent,
            "kl_scene": kl_scene,
            "pred":     pred,
        }

    # --------------------------------------------------
    # Sampling
    # --------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        x:                torch.Tensor,
        scene_embeddings: Optional[torch.Tensor] = None,
        n_samples:        int = 20,
        neighbor:         Optional[torch.Tensor] = None,
        last_obs:         Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns [n_samples, B, T, 2]. Preserves train/eval mode."""
        was_training = self.training
        self.eval()
        try:
            outs = [
                self.forward(
                    x, scene_embeddings, y=None,
                    neighbor=neighbor, last_obs=last_obs,
                )
                for _ in range(n_samples)
            ]
            return torch.stack(outs, dim=0)
        finally:
            self.train(was_training)