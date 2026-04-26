"""
Trajectory prediction evaluation metrics.

ADE: Average Displacement Error — mean L2 over all timesteps, best-of-N
FDE: Final Displacement Error   — L2 at final timestep, best-of-N
NLL: Negative Log-Likelihood    — Gaussian mixture approximation
"""

import torch
import numpy as np


def ade(pred, gt):
    """
    pred: (B, N, pred_len, 2)
    gt:   (B, pred_len, 2)
    Returns mean best-of-N ADE across batch.
    """
    gt_exp = gt.unsqueeze(1).expand_as(pred)
    l2     = torch.norm(pred - gt_exp, dim=-1).mean(dim=-1)  # (B, N)
    return l2.min(dim=1).values.mean().item()


def fde(pred, gt):
    """
    pred: (B, N, pred_len, 2)
    gt:   (B, pred_len, 2)
    Returns mean best-of-N FDE across batch.
    """
    gt_final   = gt[:, -1:, :]                               # (B, 1, 2)
    pred_final = pred[:, :, -1, :]                           # (B, N, 2)
    l2 = torch.norm(pred_final - gt_final.squeeze(1).unsqueeze(1), dim=-1)  # (B, N)
    return l2.min(dim=1).values.mean().item()


def gaussian_nll(pred, gt, sigma=1.0):
    """
    Approximate NLL treating each sample as a Gaussian component.
    pred: (B, N, pred_len, 2)
    gt:   (B, pred_len, 2)
    """
    gt_exp  = gt.unsqueeze(1).expand_as(pred)
    sq_err  = ((pred - gt_exp) ** 2).sum(dim=-1).mean(dim=-1)  # (B, N)
    log_p   = -0.5 * sq_err / (sigma ** 2)
    log_mix = torch.logsumexp(log_p, dim=1) - np.log(pred.shape[1])
    return -log_mix.mean().item()


def evaluate(model, loader, device, num_samples=20):
    model.eval()
    ade_sum = fde_sum = nll_sum = 0.0
    n = 0

    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            pred_samples = model.predict(batch, num_samples=num_samples)  # (B, N, T, 2)
            gt = batch["pred"]

            ade_sum += ade(pred_samples, gt)
            fde_sum += fde(pred_samples, gt)
            nll_sum += gaussian_nll(pred_samples, gt)
            n += 1

    return {"ADE": ade_sum / n, "FDE": fde_sum / n, "NLL": nll_sum / n}
