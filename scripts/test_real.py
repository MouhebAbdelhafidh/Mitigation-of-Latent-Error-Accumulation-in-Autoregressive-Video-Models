import argparse
from pathlib import Path
import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from lntr.data.dataset import LatentSequenceDataset
from lntr.models.forward_predictor import CausalLatentPredictor


def parse_args():
    p = argparse.ArgumentParser(description="Real-World Test Dataset Inference")
    p.add_argument(
        "--h5_test",
        type=str,
        default="cached_features/vjepa2_vitl_f32_r256/test_latents.h5",
        help="Path to your test dataset HDF5 file.",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/lntr_v2/lntr_full_best.pt",
        help="Path to your trained LN-TR production weights.",
    )
    p.add_argument("--context_steps", type=int, default=4, help="Context frames to seed rollout.")
    p.add_argument("--batch_size", type=int, default=32)
    return p.parse_args()


@torch.no_grad()
def run_real_test(model, dataloader, context_steps, device):
    model.eval()
    total_mse = 0.0
    total_samples = 0

    print("\n--- Running Real-World Test Inference ---")
    for batch in tqdm(dataloader, desc="Evaluating Test Clips"):
        batch = batch.to(device)  # [B, T, D]
        B, T, D = batch.shape
        rollout_steps = T - context_steps
        
        if rollout_steps <= 0:
            continue

        # Seed with context
        current_seq = batch[:, :context_steps, :].clone()
        batch_mse = 0.0

        for step in range(rollout_steps):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                preds = model(current_seq)
            
            next_pred = preds[:, -1:, :].float()
            gt_target = batch[:, context_steps + step : context_steps + step + 1, :]

            # Measure error against real-world test targets
            step_mse = F.mse_loss(next_pred, gt_target, reduction="mean").item()
            batch_mse += step_mse

            # Autoregressive feedback loop
            current_seq = torch.cat([current_seq, next_pred], dim=1)

        total_mse += (batch_mse / rollout_steps) * B
        total_samples += B

    mean_test_mse = total_mse / total_samples
    print(f"\n[★] Real-World Test Dataset Average Autoregressive MSE: {mean_test_mse:.4f}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = Path(__file__).resolve().parent.parent

    h5_test_path = project_root / args.h5_test
    ckpt_path = project_root / args.checkpoint

    # Load model and standardization stats from checkpoint
    print(f"[+] Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model_args = ckpt.get("args", {})
    
    model = CausalLatentPredictor(
        input_dim=1024,
        hidden_dim=model_args.get("hidden_dim", 512),
        num_layers=model_args.get("num_layers", 6),
        num_heads=model_args.get("num_heads", 8),
    ).to(device)
    model.load_state_dict(ckpt["model_forward_state_dict"])
    
    mean = ckpt["latent_mean"]
    std = ckpt["latent_std"]

    # Load test dataset
    with h5py.File(h5_test_path, "r") as f:
        seq_len = f[list(f.keys())[0]].shape[0]

    test_dataset = LatentSequenceDataset(h5_test_path, sequence_length=seq_len, mean=mean, std=std)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"[+] Loaded {len(test_dataset)} test clips from {args.h5_test}")

    # Run inference evaluation
    run_real_test(model, test_loader, args.context_steps, device)


if __name__ == "__main__":
    main()