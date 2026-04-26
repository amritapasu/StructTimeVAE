import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from struct_time_vae import StructTimeVAE
from data import Dataloader

OB_HORIZON = 4
PRED_HORIZON = 10
HIDDEN_DIM = 256

# -------------------------------------------------
# Visualization
# -------------------------------------------------
def visualize(obs, gt, preds, save_path):

    plt.figure(figsize=(6,6))

    plt.plot(obs[:,0], obs[:,1], 'bo-', label="Observed")
    plt.plot(gt[:,0], gt[:,1], 'go-', label="Ground Truth")

    for s in range(preds.shape[0]):
        plt.plot(preds[s,:,0], preds[s,:,1], color="red", alpha=0.3)

    plt.legend()
    plt.axis("equal")
    plt.tight_layout()

    plt.savefig(save_path)
    plt.close()


# -------------------------------------------------
# Sampling
# -------------------------------------------------
def sample(model, x, map_tensor, map_names, n_samples):

    samples = []

    for _ in range(n_samples):
        pred = model(x, map_tensor, map_names)
        samples.append(pred)

    return torch.stack(samples, dim=0)


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--test", nargs="+", required=True)
    parser.add_argument("--pred_samples", type=int, default=20)
    parser.add_argument("--out_dir", default="visualizations")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------
    # Load dataset
    # -------------------------------------------------
    test_dataset = Dataloader(
        args.test,
        ob_horizon=OB_HORIZON,        # observation length
        pred_horizon=PRED_HORIZON,     # prediction length
        batch_size=128,      # can be any value for visualization
        batch_first=False,
        device="cuda:0",
        seed=1,
        shuffle=False
    )

    test_data = torch.utils.data.DataLoader(
        test_dataset,
        collate_fn=test_dataset.collate_fn,
        batch_sampler=test_dataset.batch_sampler
    )

    # -------------------------------------------------
    # Load model
    # -------------------------------------------------
    model = StructTimeVAE(horizon=PRED_HORIZON, hidden_dim=HIDDEN_DIM)

    checkpoint = torch.load(args.ckpt, map_location=device)

    model.load_state_dict(checkpoint["model"])

    model = model.to(device)

    model.eval()

    print("Loaded checkpoint:", args.ckpt)

    map_cache = {}

    scene_id = 0

    # -------------------------------------------------
    # Prediction loop
    # -------------------------------------------------
    # -------------------------------------------------
    # Prediction & Visualization Loop
    # -------------------------------------------------
    scene_id = 0
    map_cache = {}

    model.eval()
    with torch.no_grad():
        for item in test_data:

            x, y, neighbor, map_names, seq_len = item

            x = x.to(device)
            y = y.to(device)
            neighbor = neighbor.to(device)

            # -------------------------
            # Ensure batch dimension
            # -------------------------
            if x.ndim == 2:       # [horizon, 2] -> [1, horizon, 2]
                x = x.unsqueeze(0)
            if y.ndim == 2:       # [horizon, 2] -> [1, horizon, 2]
                y = y.unsqueeze(0)

            B = x.shape[0]

            # -------------------------
            # Build map tensor
            # -------------------------
            batch_maps = []
            for name in map_names:
                if name not in map_cache:
                    m = test_data.dataset.map[name]
                    if m.max() > 1.0:
                        m = m / 255.0
                    map_cache[name] = m.to(device)
                batch_maps.append(map_cache[name])

            map_tensor = torch.stack(batch_maps, dim=0)  # [B, C, H, W]

            # -------------------------
            # Sample predictions
            # -------------------------
            preds = model.sample(x, map_tensor, map_names, n_samples=args.pred_samples)
            # preds shape: [S, B, horizon, 2]

            # If single batch, ensure shape is correct
            if preds.ndim == 3:
                preds = preds.unsqueeze(1)  # [S, 1, horizon, 2]

            # -------------------------
            # Visualization per scene
            # -------------------------
            x_np = x.cpu().numpy()
            y_np = y.cpu().numpy()
            preds_np = preds.cpu().numpy()

            # dataset returns [agents, batch, horizon, 2]
            if y_np.ndim == 4:
                y_np = y_np[0]   # keep primary agent

            B = x_np.shape[0]

            for b in range(B):

                obs = x_np[b]        # [obs_horizon, 2]
                gt = y_np[b]         # [pred_horizon, 2]
                pred = preds_np[:, b]  # [samples, horizon, 2]

                save_path = os.path.join(
                    args.out_dir,
                    f"scene_{scene_id:06d}.png"
                )

                visualize(obs, gt, pred, save_path)

                scene_id += 1

    print(f"Saved {scene_id} visualizations to {args.out_dir}")


if __name__ == "__main__":
    main()