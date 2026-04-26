import torch
import argparse
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data")); 
from sdd_dataset import build_dataloaders
from models.context_vae import ContextVAE
from utils.metrics import evaluate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",   type=str, default="data")
    parser.add_argument("--checkpoint",  type=str, default="checkpoints/best_early.pt")
    parser.add_argument("--map_mode",    type=str, default="early")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--batch_size",  type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, test_loader, _ = build_dataloaders(
        root_dir=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    model = ContextVAE(map_mode=args.map_mode).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
    print(f"Loaded checkpoint: {args.checkpoint}")

    metrics = evaluate(model, test_loader, device, num_samples=args.num_samples)
    print(f"\nResults (best-of-{args.num_samples}):")
    print(f"  ADE: {metrics['ADE']:.4f}")
    print(f"  FDE: {metrics['FDE']:.4f}")
    print(f"  NLL: {metrics['NLL']:.4f}")

if __name__ == "__main__":
    main()
