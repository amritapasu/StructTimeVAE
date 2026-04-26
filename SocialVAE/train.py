"""
Training script for ContextVAE baseline on SDD.

Usage:
    python train.py --data_root /path/to/sdd --map_mode early
    python train.py --data_root /path/to/sdd --map_mode none
    python train.py --data_root /path/to/sdd --map_mode late

Map ablation modes:
    none  : encoder sees only agent + neighbors (no map)
    early : map fused into VAE encoder (default ContextVAE)
    late  : map concatenated to Z before decoder
"""

import argparse
import os
import json
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data.sdd_dataset import build_dataloaders
from models.context_vae import ContextVAE
from utils.metrics import evaluate


def kl_loss(mu, logvar):
    """KL divergence from N(mu, exp(logvar)) to N(0, 1)."""
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def train_epoch(model, loader, optimizer, device, beta):
    model.train()
    total_loss = total_rec = total_kl = 0.0

    for batch in loader:

        x, y, neighbor, map_names, seq_len = batch
        print("x:", x.shape)       # expect [obs_len, B, input_dim]
        print("y:", y.shape)       # expect [B, pred_len, 2]
        print("map_names:", map_names[:3])

        print(f"rec={loss_dict['rec'].item():.4f}  "
        f"kl_agent={loss_dict['kl_agent'].item():.4f}  "
        f"kl_scene={loss_dict['kl_scene'].item():.4f}")
        break
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        optimizer.zero_grad()

        pred_rel, pred_abs, mu, logvar = model(batch)

        rec  = F.mse_loss(pred_rel, batch["pred_rel"])
        kl   = kl_loss(mu, logvar)
        loss = rec + beta * kl

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_rec  += rec.item()
        total_kl   += kl.item()

    n = len(loader)
    return total_loss / n, total_rec / n, total_kl / n


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  map_mode: {args.map_mode}")

    train_loader, test_loader, _ = build_dataloaders(
        args.data_root,
        batch_size=args.batch_size,
        use_map=(args.map_mode != 'none'),
        num_workers=args.num_workers,
    )

    model = ContextVAE(
        map_mode=args.map_mode,
        latent_dim=args.latent_dim,
        num_samples=args.num_samples,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    os.makedirs(args.save_dir, exist_ok=True)
    best_ade = float("inf")
    history  = []

    for epoch in range(1, args.epochs + 1):
        # KL annealing: ramp beta from 0 → 1 over first 20 epochs
        beta = min(1.0, epoch / 20.0)

        loss, rec, kl = train_epoch(model, train_loader, optimizer, device, beta)
        log = {"epoch": epoch, "loss": loss, "rec": rec, "kl": kl}

        if epoch % args.eval_every == 0:
            metrics = evaluate(model, test_loader, device, args.num_samples)
            log.update(metrics)
            scheduler.step(metrics["ADE"])

            print(
                f"Epoch {epoch:03d} | loss={loss:.4f} rec={rec:.4f} kl={kl:.4f} "
                f"| ADE={metrics['ADE']:.4f} FDE={metrics['FDE']:.4f} NLL={metrics['NLL']:.4f}"
            )

            if metrics["ADE"] < best_ade:
                best_ade = metrics["ADE"]
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "metrics": metrics,
                    "args": vars(args),
                }, os.path.join(args.save_dir, f"best_{args.map_mode}.pt"))
                print(f"  -> Saved best (ADE={best_ade:.4f})")
        else:
            print(f"Epoch {epoch:03d} | loss={loss:.4f} rec={rec:.4f} kl={kl:.4f}")

        history.append(log)

    with open(os.path.join(args.save_dir, f"history_{args.map_mode}.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nFinished. Best ADE: {best_ade:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",   required=True)
    parser.add_argument("--map_mode",    default="early", choices=["none", "early", "late"])
    parser.add_argument("--epochs",      type=int,   default=100)
    parser.add_argument("--batch_size",  type=int,   default=64)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--latent_dim",  type=int,   default=32)
    parser.add_argument("--num_samples", type=int,   default=20)
    parser.add_argument("--eval_every",  type=int,   default=5)
    parser.add_argument("--num_workers", type=int,   default=4)
    parser.add_argument("--save_dir",    default="checkpoints")
    args = parser.parse_args()
    main(args)
