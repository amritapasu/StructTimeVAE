from typing import Optional, List, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class StructTimeVAE(nn.Module):
    """
    StructTimeVAE with social attention:

    Architecture:
      - z_agent : stochastic CVAE latent capturing agent intent uncertainty
                  prior   p(z_agent | h_agent, h_social, h_scene)
                  posterior q(z_agent | h_agent, h_social, h_scene, h_future)
      - h_social : deterministic social context from neighbor attention
                  SocialAttention(target_state, neighbor_states) -> h_social
      - scene context : deterministic, temporally-evolving
                  h_scene = DINOv2(map) -> scene_proj -> scene_norm
                  c_scene(t) = GRU(h_scene)[t]

    Neighbor tensor convention (from dataloader):
      neighbor : [obs_len, B, N_max, 6]
        - features: [abs_x, abs_y, vel_x, vel_y, acc_x, acc_y]
        - padding:  1e9 sentinel for missing neighbors
        - relative features are computed internally (neighbor - target)

    Input conventions:
      x          : [obs_len, B, input_dim]
      neighbor   : [obs_len, B, N_max, 6]  or None
      map_tensor : [num_unique_maps, 3, H, W]
      map_names  : list[str] of length B
      y          : [B, pred_len, 2]  (training only)
    """

    PADDING_SENTINEL = 1e8   # neighbor values >= this are padding

    # --------------------------------------------------
    # SocialAttention
    # --------------------------------------------------

    class SocialAttention(nn.Module):
        """
        Encodes neighbor context via scaled dot-product attention.

        For each agent i at each observation timestep t:
          - Compute relative features of each neighbor j:
              rel_pos  = neighbor_pos_j - target_pos_i   [2]
              rel_vel  = neighbor_vel_j - target_vel_i   [2]
              distance = ||rel_pos||                      [1]
              rel_feat = concat(rel_pos, rel_vel, dist)  [5]
          - Embed rel_feat -> key/value vectors per neighbor
          - Use target agent's GRU hidden state as query
          - Masked softmax attention over real neighbors (padding excluded)
          - Pool to single social context vector h_social

        Then aggregate over observation timesteps by taking the
        final timestep's social context (most recent observation).
        """
        def __init__(self, hidden_dim: int, neighbor_feat_dim: int = 5):
            super().__init__()
            self.hidden_dim = hidden_dim

            # Embed relative neighbor features to key and value
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
            # Project query (target hidden state) to attention space
            self.query_proj = nn.Linear(hidden_dim, hidden_dim)

            # Output projection
            self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        def forward(
            self,
            h_agent:  torch.Tensor,   # [B, hidden_dim]
            x:        torch.Tensor,   # [obs_len, B, 6]
            neighbor: torch.Tensor,   # [obs_len, B, N, 6]
        ) -> torch.Tensor:
            x_last = x[-1]        # [B, 6]
            n_last = neighbor[-1] # [B, N, 6]

            B, N, F = n_last.shape

            # Padding mask: True = real neighbor
            pad_mask = n_last[:, :, 0] < StructTimeVAE.PADDING_SENTINEL  # [B, N]

            # Zero out padded entries before computing relative features
            n_clamped = n_last * pad_mask.unsqueeze(-1).float()

            target_pos = x_last[:, :2].unsqueeze(1)   # [B, 1, 2]
            target_vel = x_last[:, 2:4].unsqueeze(1)  # [B, 1, 2]

            rel_pos  = n_clamped[:, :, :2] - target_pos   # [B, N, 2]
            rel_vel  = n_clamped[:, :, 2:4] - target_vel  # [B, N, 2]
            distance = rel_pos.norm(dim=-1, keepdim=True)  # [B, N, 1]

            rel_feat = torch.cat([rel_pos, rel_vel, distance], dim=-1)  # [B, N, 5]

            # Zero padded entries in rel_feat (out-of-place)
            rel_feat = rel_feat * pad_mask.unsqueeze(-1).float()

            # Keys and values
            keys   = self.key_embed(rel_feat)   # [B, N, hidden_dim]
            values = self.val_embed(rel_feat)   # [B, N, hidden_dim]

            # Query
            query = self.query_proj(h_agent).unsqueeze(2)  # [B, hidden_dim, 1]

            # Attention scores
            scale  = self.hidden_dim ** 0.5
            scores = torch.bmm(keys, query).squeeze(2) / scale  # [B, N]

            # Mask padding (out-of-place)
            pad_scores = scores.masked_fill(~pad_mask, float('-inf'))

            # Handle agents with no real neighbors: avoid all-inf softmax
            all_padded  = ~pad_mask.any(dim=-1)                              # [B]
            safe_scores = pad_scores.masked_fill(
                all_padded.unsqueeze(1).expand_as(pad_scores), 0.0
            )

            attn = torch.softmax(safe_scores, dim=-1)                        # [B, N]

            # Zero attention for all-padded agents (out-of-place)
            attn = attn.masked_fill(all_padded.unsqueeze(1), 0.0)

            # Weighted sum
            h_social = torch.bmm(attn.unsqueeze(1), values).squeeze(1)      # [B, hidden_dim]
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
        horizon:       int,
        input_dim:     int   = 6,
        hidden_dim:    int   = 256,
        z_agent_dim:   int   = 32,
        scene_dim:     int   = 32,
        kl_beta_agent: float = 1.0,
        kl_beta_scene: float = 1.0,   # kept for API compatibility
        free_bits:     float = 0.01,
        dinov2_path:   Optional[str] = None,
        device:        str   = "cuda:0",
    ):
        super().__init__()

        self.horizon       = horizon
        self.input_dim     = input_dim
        self.hidden_dim    = hidden_dim
        self.z_agent_dim   = z_agent_dim
        self.scene_dim     = scene_dim
        self.kl_beta_agent = kl_beta_agent
        self.free_bits     = free_bits
        self.device        = torch.device(device)

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
        # Future encoder (posterior, training only)
        # --------------------------
        self.future_embed = nn.Linear(2, hidden_dim)
        self.future_rnn   = nn.GRU(hidden_dim, hidden_dim)

        # --------------------------
        # Frozen DINOv2 scene encoder
        # --------------------------
        if dinov2_path is not None:
            self.scene_encoder = torch.load(dinov2_path, map_location="cpu")
        else:
            self.scene_encoder = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vits14",
            )
        self.scene_encoder.eval()
        for p in self.scene_encoder.parameters():
            p.requires_grad = False

        self.scene_proj = nn.Sequential(
            nn.Linear(384, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.scene_norm = nn.LayerNorm(hidden_dim)

        # --------------------------
        # Temporal scene GRU
        # --------------------------
        self.scene_h0_proj  = nn.Sequential(
            nn.Linear(hidden_dim, scene_dim),
            nn.Tanh(),
        )
        self.scene_time_gru = nn.GRU(1, scene_dim)

        # --------------------------
        # z_agent latent networks
        #
        # prior_ctx:  [h_agent, h_social, h_scene]       = 3H
        # post_ctx:   [h_agent, h_social, h_scene, h_fut] = 4H
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
        mu_q:     torch.Tensor,
        logvar_q: torch.Tensor,
        mu_p:     torch.Tensor,
        logvar_p: torch.Tensor,
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
        h_agent:  torch.Tensor,            # [B, hidden_dim]
        x:        torch.Tensor,            # [obs_len, B, 6]  displacement space
        neighbor: Optional[torch.Tensor],  # [obs_len, B, N, 6] or None
        last_obs: Optional[torch.Tensor],  # [B, 2] last absolute position
    ) -> torch.Tensor:                     # [B, hidden_dim]
        """
        Computes social context vector h_social.

        If neighbor is None or all padding, returns zeros.

        Note: neighbor features from the dataloader are in absolute pixel
        coordinates. We convert to displacement space before attention
        by subtracting last_obs (the target agent's last absolute position).
        This keeps neighbor encoding consistent with the displacement-space
        x that the agent encoder sees.
        """
        if neighbor is None:
            return torch.zeros(h_agent.size(0), self.hidden_dim, device=self.device)

        neighbor = neighbor.to(self.device, non_blocking=True)
        x_dev    = x.to(self.device, non_blocking=True)

        # neighbor is in absolute coords; convert positions to relative
        # by subtracting target agent's last observed absolute position
        if last_obs is not None:
            last_obs = last_obs.to(self.device, non_blocking=True)
            # Only shift position features (first 2), not velocity/accel
            # Mask padding before shifting
            pad_mask = neighbor[:, :, :, 0] < self.PADDING_SENTINEL  # [T, B, N]
            # With:
            pos_shift  = last_obs.unsqueeze(0).unsqueeze(2)           # [1, B, 1, 2]
            shifted_pos = neighbor[:, :, :, :2] - pos_shift           # [T, B, N, 2]
            neighbor   = torch.cat([shifted_pos, neighbor[:, :, :, 2:]], dim=-1)
            # Restore padding sentinel out-of-place
            sentinel   = torch.full_like(neighbor, self.PADDING_SENTINEL)
            pad_mask_4 = (neighbor[:, :, :, 0] < self.PADDING_SENTINEL).unsqueeze(-1).expand_as(neighbor)
            neighbor   = torch.where(pad_mask_4, neighbor, sentinel)

        return self.social_attn(h_agent, x_dev, neighbor)

    def encode_future(self, y: torch.Tensor) -> torch.Tensor:
        """y: [pred_len, B, 2] time-first -> [B, hidden_dim]. Training only."""
        y = y.to(self.device, non_blocking=True)
        T, B, D = y.shape
        y_emb = self.future_embed(y.reshape(T * B, D)).view(T, B, self.hidden_dim)
        _, h  = self.future_rnn(y_emb)
        return h.squeeze(0)

    def encode_scene_batch(self, map_tensor: torch.Tensor) -> torch.Tensor:
        """map_tensor: [N, 3, H, W] -> [N, hidden_dim]"""
        map_tensor = map_tensor.to(self.device, dtype=torch.float32, non_blocking=True)
        if map_tensor.dim() != 4 or map_tensor.size(1) != 3:
            raise ValueError(f"Expected [N,3,H,W], got {map_tensor.shape}")

        N, C, H, W = map_tensor.shape
        scale    = 224.0 / min(H, W)
        new_h, new_w = int(round(H * scale)), int(round(W * scale))
        imgs = F.interpolate(map_tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)

        top  = (new_h - 224) // 2
        left = (new_w - 224) // 2
        imgs = imgs[:, :, top:top + 224, left:left + 224]

        if imgs.size(-2) != 224 or imgs.size(-1) != 224:
            raise ValueError(f"Center crop produced {imgs.shape[-2:]}, expected 224x224")

        if map_tensor.max().item() > 2.0:
            imgs = imgs / 255.0

        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        imgs = (imgs - mean) / std

        with torch.no_grad():
            feat = self.scene_encoder(imgs)

        return self.scene_norm(self.scene_proj(feat))

    def assign_scene_embeddings(
        self,
        map_tensor: Optional[torch.Tensor],
        map_names:  Optional[List[str]],
        batch_size: int,
    ) -> torch.Tensor:
        """Returns per-agent scene embeddings [B, hidden_dim]."""
        if map_tensor is None:
            return torch.zeros(batch_size, self.hidden_dim, device=self.device)

        if map_names is None:
            h = self.encode_scene_batch(map_tensor)
            if h.size(0) == 1:
                return h.expand(batch_size, -1)
            if h.size(0) == batch_size:
                return h
            raise ValueError(
                f"map_names=None but map_tensor has {h.size(0)} entries "
                f"and batch_size={batch_size}."
            )

        unique_names = list(dict.fromkeys(map_names))
        if len(unique_names) != map_tensor.size(0):
            raise ValueError(
                f"{len(unique_names)} unique map names but map_tensor has "
                f"{map_tensor.size(0)} entries."
            )

        unique_h  = self.encode_scene_batch(map_tensor)
        name_to_h: Dict[str, torch.Tensor] = {
            name: unique_h[i] for i, name in enumerate(unique_names)
        }
        return torch.stack([name_to_h[n] for n in map_names], dim=0)

    # --------------------------------------------------
    # Scene context temporal evolution
    # --------------------------------------------------

    def evolve_scene_context(self, h_scene: torch.Tensor) -> torch.Tensor:
        """
        h_scene: [B, hidden_dim] -> [B, horizon, scene_dim]
        GRU receives zero dummy inputs; h_scene seeds hidden state.
        """
        B     = h_scene.size(0)
        h     = self.scene_h0_proj(h_scene).unsqueeze(0)
        dummy = torch.zeros(1, B, 1, device=h_scene.device)

        seq = []
        for _ in range(self.horizon):
            out, h = self.scene_time_gru(dummy, h)
            seq.append(out.squeeze(0))

        return torch.stack(seq, dim=1)   # [B, horizon, scene_dim]

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------

    def forward(
        self,
        x:             torch.Tensor,
        map_tensor:    Optional[torch.Tensor],
        map_names:     Optional[List[str]] = None,
        y:             Optional[torch.Tensor] = None,
        neighbor:      Optional[torch.Tensor] = None,
        last_obs:      Optional[torch.Tensor] = None,
        kl_beta_agent: Optional[float] = None,
        kl_beta_scene: Optional[float] = None,
    ):
        """
        Inference (y=None):  returns pred [B, T, 2]
        Training  (y given): returns dict: loss, rec, kl_agent, kl_scene, pred

        neighbor: [obs_len, B, N, 6] raw absolute coords from dataloader, or None
        last_obs: [B, 2] last observed absolute position (for neighbor relativization)
                  Pass this from main.py after computing to_displacements.
        """
        x       = x.to(self.device, non_blocking=True)
        h_agent = self.encode_agent(x)
        B       = h_agent.size(0)

        # Social context
        h_social = self.encode_social(h_agent, x, neighbor, last_obs)  # [B, hidden_dim]

        # Scene context
        h_scene     = self.assign_scene_embeddings(map_tensor, map_names, B)
        c_scene_seq = self.evolve_scene_context(h_scene)               # [B, T, scene_dim]

        # Prior: conditioned on agent + social + scene
        prior_ctx          = torch.cat([h_agent, h_social, h_scene], dim=-1)  # [B, 3H]
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

        post_ctx           = torch.cat([h_agent, h_social, h_scene, h_future], dim=-1)  # [B, 4H]
        mu_q_a, logvar_q_a = self.q_z_agent(post_ctx)

        z_agent  = self._reparameterize(mu_q_a, logvar_q_a)
        pred     = self.decoder(h_agent, z_agent, c_scene_seq)

        rec      = F.mse_loss(pred, y)
        kl_agent = self._kl_gaussian(mu_q_a, logvar_q_a, mu_p_a, logvar_p_a, self.free_bits)
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
        x:          torch.Tensor,
        map_tensor: Optional[torch.Tensor],
        map_names:  Optional[List[str]],
        n_samples:  int = 20,
        neighbor:   Optional[torch.Tensor] = None,
        last_obs:   Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns: [n_samples, B, T, 2]
        """
        was_training = self.training
        self.eval()
        try:
            outs = [
                self.forward(
                    x, map_tensor, map_names,
                    y=None, neighbor=neighbor, last_obs=last_obs
                )
                for _ in range(n_samples)
            ]
            return torch.stack(outs, dim=0)
        finally:
            self.train(was_training)