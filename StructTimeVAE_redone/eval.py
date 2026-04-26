# eval.py
# Runs evaluation on the test set only.
# Usage:
#   python3 /p3/home/mekeys/StructTimeVAE_redone/eval.py \
#       --test /p3/home/mekeys/SDD_embeddings/test \
#       --config /p3/home/mekeys/StructTimeVAE_redone/config/sdd_eval.py \
#       --ckpt /p3/home/mekeys/StructTimeVAE_redone/checkpoints/best_ADE.pt \
#       --cache_path /p3/home/mekeys/SDD_embeddings/cache.h5 \
#       --device cuda:0

import os, sys
import torch
import numpy as np

from struct_time_vae import StructTimeVAE
from data import Dataloader
from utils import ADE_FDE, seed

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--test",       nargs="+", required=True,  help="Path to test data directory")
parser.add_argument("--config",     type=str,  required=True,  help="Path to config file")
parser.add_argument("--ckpt",       type=str,  required=True,  help="Path to checkpoint .pt file")
parser.add_argument("--cache_path", type=str,  default=None,   help="Path to h5 embedding cache")
parser.add_argument("--device",     type=str,  default=None,   help="Device (default: cuda if available)")
parser.add_argument("--seed",       type=int,  default=1)

if __name__ == "__main__":
    settings = parser.parse_args()

    import importlib
    spec = importlib.util.spec_from_file_location(
        "config", settings.config,
        submodule_search_locations=[os.path.dirname(settings.config)]
    )
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    device = settings.device if settings.device else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    seed(settings.seed)

    # -----------------------------
    # Data
    # -----------------------------
    test_dataset = Dataloader(
        settings.test,
        batch_first=False,
        device=device,
        seed=settings.seed,
        shuffle=False,
        cache_path=settings.cache_path,
        **config.test_dataloader,
    )
    test_data = torch.utils.data.DataLoader(
        test_dataset,
        collate_fn=test_dataset.collate_fn,
        batch_sampler=test_dataset.batch_sampler,
    )

    # -----------------------------
    # Model
    # -----------------------------
    model = StructTimeVAE(**config.model)
    model.to(device)

    print(f"Loading checkpoint: {settings.ckpt}")
    state_dict = torch.load(settings.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state_dict["model"])
    print(f"Loaded checkpoint from epoch {state_dict.get('epoch', '?')}")

    # -----------------------------
    # Evaluate
    # -----------------------------
    def evaluate(dataloader):
        ADE, FDE, ADE_d, FDE_d = [], [], [], []
        n_samples = getattr(config, "pred_samples", 50)
        with torch.no_grad():
            for item in dataloader:
                if dataloader.dataset.use_map:
                    x, y, neighbor, scene_embeddings, seq_len = item
                    scene_embeddings = scene_embeddings.to(device)
                else:
                    x, y, neighbor, seq_len = item
                    scene_embeddings = None

                x        = x.to(device)
                y        = y.to(device)
                neighbor = neighbor.to(device)

                if y.dim() != 3:
                    y = y.reshape(-1, y.shape[-2], y.shape[-1])

                # stochastic: min over n_samples
                y_samples = model.sample(x, scene_embeddings, n_samples=n_samples)
                ade_b, fde_b = ADE_FDE(y_samples, y)
                ADE.append(torch.min(ade_b, dim=0)[0])
                FDE.append(torch.min(fde_b, dim=0)[0])

                # deterministic
                y_det = model(x, scene_embeddings)
                ade_d, fde_d = ADE_FDE(y_det.unsqueeze(0), y)
                ADE_d.append(ade_d.squeeze(0))
                FDE_d.append(fde_d.squeeze(0))

        return (
            torch.cat(ADE).mean().item(),
            torch.cat(FDE).mean().item(),
            torch.cat(ADE_d).mean().item(),
            torch.cat(FDE_d).mean().item(),
        )

    model.eval()
    ade, fde, ade_d, fde_d = evaluate(test_data)

    print(f"\nTest Results")
    print(f"  Stochastic    — ADE: {ade:.4f}  FDE: {fde:.4f}")
    print(f"  Deterministic — ADE: {ade_d:.4f}  FDE: {fde_d:.4f}")