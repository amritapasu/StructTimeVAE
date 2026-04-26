"""
ContextVAE Baseline
===================
Single fused latent Z encodes agent past + social context + scene map.
This is the entangled baseline the factorized model will be compared against.

Architecture:
    AgentEncoder  : LSTM over agent's past velocity sequence
    SocialEncoder : LSTM over each neighbor's past, mean-pooled
    MapEncoder    : CNN over rasterized scene patch
    VAEEncoder    : fused context -> (mu, logvar) -> Z
    Decoder       : non-autoregressive MLP(Z, last_obs_pos) -> future deltas

Batch dict keys (from sdd_dataset.py):
    obs           (B, obs_len, 2)
    pred          (B, pred_len, 2)
    obs_rel       (B, obs_len, 2)
    pred_rel      (B, pred_len, 2)
    neighbors     (B, max_neighbors, obs_len, 2)
    num_neighbors (B,)  int list
    patch         (B, 3, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class AgentEncoder(nn.Module):
    """LSTM over the agent's observed velocity sequence."""

    def __init__(self, input_dim=2, hidden_dim=64, out_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, obs_rel):
        # obs_rel: (B, obs_len, 2)
        _, (h, _) = self.lstm(obs_rel)
        return F.relu(self.proj(h.squeeze(0)))  # (B, out_dim)


class SocialEncoder(nn.Module):
    """
    LSTM over each neighbor's observed trajectory, mean-pooled across neighbors.
    Zero-padded neighbors are excluded from the mean via the num_neighbors mask.
    """

    def __init__(self, input_dim=2, hidden_dim=64, out_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, neighbors, num_neighbors):
        # neighbors:     (B, max_neighbors, obs_len, 2)
        # num_neighbors: list of ints, length B
        B, N, T, D = neighbors.shape

        # Run LSTM over all (B*N) neighbor sequences at once
        flat = neighbors.view(B * N, T, D)
        _, (h, _) = self.lstm(flat)
        h = h.squeeze(0).view(B, N, -1)   # (B, N, hidden_dim)

        # Build mask: 1 for real neighbors, 0 for padding
        mask = torch.zeros(B, N, 1, device=neighbors.device)
        for i, n in enumerate(num_neighbors):
            mask[i, :n, 0] = 1.0

        # Masked mean pool
        h_sum   = (h * mask).sum(dim=1)                    # (B, hidden_dim)
        denom   = mask.sum(dim=1).clamp(min=1.0)           # (B, 1)
        h_mean  = h_sum / denom                            # (B, hidden_dim)

        return F.relu(self.proj(h_mean))                   # (B, out_dim)


class MapEncoder(nn.Module):
    """Small CNN over a rasterized scene image patch."""

    def __init__(self, in_channels=3, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),          nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),          nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.proj = nn.Linear(64, out_dim)

    def forward(self, x):
        # x: (B, 3, H, W)
        return F.relu(self.proj(self.net(x)))  # (B, out_dim)


class VAEEncoder(nn.Module):
    """Projects fused context to (mu, logvar) for the latent Z."""

    def __init__(self, context_dim, latent_dim, hidden_dim=128):
        super().__init__()
        self.fc      = nn.Linear(context_dim, hidden_dim)
        self.fc_mu   = nn.Linear(hidden_dim, latent_dim)
        self.fc_lv   = nn.Linear(hidden_dim, latent_dim)

    def forward(self, context):
        h      = F.relu(self.fc(context))
        mu     = self.fc_mu(h)
        logvar = self.fc_lv(h)
        return mu, logvar


class TrajectoryDecoder(nn.Module):
    """
    Non-autoregressive decoder.
    Input:  Z concatenated with the agent's last observed position
    Output: pred_len velocity deltas -> cumsum to absolute positions
    """

    def __init__(self, latent_dim, pred_len=12, hidden_dim=128):
        super().__init__()
        self.pred_len = pred_len
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),     nn.ReLU(),
            nn.Linear(hidden_dim, pred_len * 2),
        )

    def forward(self, z, last_obs):
        # z:        (B, latent_dim)
        # last_obs: (B, 2)
        out      = self.net(torch.cat([z, last_obs], dim=-1))  # (B, pred_len*2)
        deltas   = out.view(-1, self.pred_len, 2)              # (B, pred_len, 2)

        # Integrate deltas from last observed position
        positions = torch.zeros_like(deltas)
        positions[:, 0] = last_obs + deltas[:, 0]
        for t in range(1, self.pred_len):
            positions[:, t] = positions[:, t - 1] + deltas[:, t]

        return deltas, positions   # (B, pred_len, 2) each


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

def reparameterize(mu, logvar):
    if mu.requires_grad or (hasattr(mu, 'grad_fn') and mu.grad_fn is not None):
        # Training mode check via gradient context is unreliable;
        # always sample — controlled by torch.no_grad() at inference.
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)
    return mu


class ContextVAE(nn.Module):
    """
    ContextVAE baseline: single latent Z entangles agent, social, and map context.

    map_mode:
        'none'  — map encoder output zeroed; Z encodes only agent + social
        'early' — map fused into VAE encoder input (default)
        'late'  — map bypasses encoder, concatenated to Z before decoder
    """

    def __init__(self, obs_len=8, pred_len=12, latent_dim=32,
                 agent_dim=64, social_dim=64, map_dim=64,
                 decoder_hidden=128, num_samples=20,
                 map_mode='early'):
        super().__init__()
        self.obs_len     = obs_len
        self.pred_len    = pred_len
        self.latent_dim  = latent_dim
        self.map_mode    = map_mode
        self.num_samples = num_samples
        self._map_dim    = map_dim

        self.agent_enc  = AgentEncoder(out_dim=agent_dim)
        self.social_enc = SocialEncoder(out_dim=social_dim)
        self.map_enc    = MapEncoder(out_dim=map_dim)

        # VAE encoder input dimension depends on map_mode
        enc_input_dim = agent_dim + social_dim + (map_dim if map_mode == 'early' else 0)
        self.vae_enc  = VAEEncoder(enc_input_dim, latent_dim, decoder_hidden)

        # Decoder input dimension depends on map_mode
        dec_input_dim = latent_dim + (map_dim if map_mode == 'late' else 0)
        self.decoder  = TrajectoryDecoder(dec_input_dim, pred_len, decoder_hidden)

    def _encode_context(self, batch):
        h_agent  = self.agent_enc(batch["obs_rel"])
        h_social = self.social_enc(batch["neighbors"], batch["num_neighbors"])
        h_map    = self.map_enc(batch["patch"])

        if self.map_mode == 'early':
            context = torch.cat([h_agent, h_social, h_map], dim=-1)
        elif self.map_mode == 'none':
            h_map   = torch.zeros_like(h_map)
            context = torch.cat([h_agent, h_social], dim=-1)
        else:  # late
            context = torch.cat([h_agent, h_social], dim=-1)

        mu, logvar = self.vae_enc(context)
        return mu, logvar, h_map

    def forward(self, batch):
        """
        Training forward pass.
        Returns:
            pred_rel:  (B, pred_len, 2)   predicted velocity deltas
            pred_abs:  (B, pred_len, 2)   predicted absolute positions
            mu, logvar: for KL loss
        """
        mu, logvar, h_map = self._encode_context(batch)
        z = reparameterize(mu, logvar)

        if self.map_mode == 'late':
            z = torch.cat([z, h_map], dim=-1)

        last_obs = batch["obs"][:, -1, :]
        pred_rel, pred_abs = self.decoder(z, last_obs)

        return pred_rel, pred_abs, mu, logvar

    @torch.no_grad()
    def predict(self, batch, num_samples=None):
        """
        Sample multiple trajectory predictions at inference.
        Returns: (B, num_samples, pred_len, 2) absolute positions
        """
        self.eval()
        n = num_samples or self.num_samples
        mu, logvar, h_map = self._encode_context(batch)
        last_obs = batch["obs"][:, -1, :]

        samples = []
        for _ in range(n):
            z = reparameterize(mu, logvar)
            if self.map_mode == 'late':
                z = torch.cat([z, h_map], dim=-1)
            _, pred_abs = self.decoder(z, last_obs)
            samples.append(pred_abs)

        return torch.stack(samples, dim=1)  # (B, n, pred_len, 2)
