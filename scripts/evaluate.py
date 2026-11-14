import argparse
from pathlib import Path
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lntr.data.dataset import LatentSequenceDataset
from lntr.models.forward_predictor import CausalLatentPredictor


def parse_args():
    p = argparse.ArgumentParser(description="Phase 4: Open-Loop Rollout Evaluation")
    p.add_argument(
        "--h5_val",
        type=str,
        default="cached_features/vjepa2_vitl_f32_r256/val_latents.h5",
        help="Path to validation latents HDF5.",
    )
    p.add_argument(
        "--lntr_ckpt",
        type=str,
        default="checkpoints/lntr_v2/lntr_full_best.pt",
        help="Path to best LN-TR checkpoint.",
    )
    p.add_argument(
        "--baseline_ckpt",
        type=str,
        default="checkpoints/baseline_v2/lntr_full_best.pt",
        help="Path to best Baseline checkpoint.",
    )
    p.add_argument(
        "--context_steps",
        type=int,
        default=4,
        help="Number of ground-truth context frames to seed the rollout.",
    )
    p.add_argument(
        "--max_rollout_steps",
        type=int,
        default=28,
        help="Maximum open-loop rollout horizon.",
    )
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--output_dir", type=str, default="outputs/evaluation")
    return p.parse_args()


def load_model_and_stats(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt.get("args", {})
    
    model = CausalLatentPredictor(
        input_dim=1024,
        hidden_dim=args.get("hidden_dim", 512),
        num_layers=args.get("num_layers", 6),
        num_heads=args.get("num_heads", 8),
    ).to(device)
    
    model.load_state_dict(ckpt["model_forward_state_dict"])
    model.eval()
    
    mean = ckpt["latent_mean"]
    std = ckpt["latent_std"]
    return model, mean, std


@torch.no_grad()
def evaluate_open_loop(model, dataloader, context_steps, max_steps, device):
    model.eval()
    step_mses = None
    step_cos_sims = None
    total_samples = 0

    for batch in tqdm(dataloader, desc="Evaluating Open-Loop Rollout"):
        batch = batch.to(device)  # [B, T, D]
        B, T, D = batch.shape
        
        # Dynamically clamp rollout horizon to available sequence length minus context
        max_possible_horizon = T - context_steps
        rollout_horizon = min(max_possible_horizon, max_steps)
        
        if rollout_horizon <= 0:
            raise ValueError(f"Sequence length T={T} must be greater than context_steps={context_steps}")

        if step_mses is None:
            step_mses = torch.zeros(rollout_horizon, device=device)
            step_cos_sims = torch.zeros(rollout_horizon, device=device)

        current_seq = batch[:, :context_steps, :].clone()

        for step in range(rollout_horizon):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                preds = model(current_seq)
            
            next_pred = preds[:, -1:, :].float()
            gt_target = batch[:, context_steps + step : context_steps + step + 1, :]

            mse = F.mse_loss(next_pred, gt_target, reduction="none").mean(dim=-1).sum()
            cos = (
                F.normalize(next_pred, p=2, dim=-1) * F.normalize(gt_target, p=2, dim=-1)
            ).sum(dim=-1).sum()

            step_mses[step] += mse
            step_cos_sims[step] += cos

            current_seq = torch.cat([current_seq, next_pred], dim=1)

        total_samples += B

    avg_mses = (step_mses / total_samples).cpu().numpy()
    avg_cos = (step_cos_sims / total_samples).cpu().numpy()
    return avg_mses, avg_cos


def plot_comparison(lntr_metrics, base_metrics, copy_last_mse, context_steps, output_dir):
    lntr_mse, lntr_cos = lntr_metrics
    base_mse, base_cos = base_metrics
    steps = np.arange(1, len(lntr_mse) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # --- Panel 1: Open-Loop MSE Drift ---
    ax1.plot(steps, base_mse, label="Control Baseline (Teacher-Forced)", color="#d95f02", linewidth=2.5, marker="s", markersize=4)
    ax1.plot(steps, lntr_mse, label="LN-TR (Ours: LSNI + RLCC)", color="#1b9e77", linewidth=2.5, marker="o", markersize=4)
    ax1.plot(steps, copy_last_mse, label="Copy-Last-Context Reference", color="gray", linestyle=":", linewidth=1.8)

    ax1.set_title(f"Open-Loop Error Accumulation (Seeded with {context_steps} Context Steps)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Autoregressive Rollout Horizon (Steps into Future)", fontsize=11)
    ax1.set_ylabel("Mean Squared Error (Standardized Space)", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Directional Alignment (Cosine Similarity) ---
    ax2.plot(steps, base_cos, label="Control Baseline (Teacher-Forced)", color="#d95f02", linewidth=2.5, marker="s", markersize=4)
    ax2.plot(steps, lntr_cos, label="LN-TR (Ours: LSNI + RLCC)", color="#1b9e77", linewidth=2.5, marker="o", markersize=4)

    ax2.set_title("Latent Trajectory Directional Fidelity (Cosine Similarity)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Autoregressive Rollout Horizon (Steps into Future)", fontsize=11)
    ax2.set_ylabel("Cosine Similarity with Ground Truth", fontsize=11)
    ax2.set_ylim(0.0, 1.05)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = output_dir / "open_loop_rollout_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[★] Saved comparison plot to: {save_path}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = Path(__file__).resolve().parent.parent

    h5_val_path = project_root / args.h5_val
    lntr_ckpt_path = project_root / args.lntr_ckpt
    base_ckpt_path = project_root / args.baseline_ckpt
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Phase 4: Open-Loop Rollout Evaluation ===")
    
    model_lntr, mean, std = load_model_and_stats(lntr_ckpt_path, device)
    model_base, _, _ = load_model_and_stats(base_ckpt_path, device)

    # Inspect HDF5 sequence length dynamically
    with h5py.File(h5_val_path, "r") as f:
        first_key = list(f.keys())[0]
        seq_len = f[first_key].shape[0]

    # Request up to 32 if available, otherwise use dataset native seq_len
    target_seq_len = max(seq_len, args.context_steps + args.max_rollout_steps)
    val_dataset = LatentSequenceDataset(h5_val_path, sequence_length=target_seq_len, mean=mean, std=std)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"[+] Loaded {len(val_dataset)} validation clips (HDF5 Native T={seq_len})")

    print("\n--- Evaluating LN-TR Model ---")
    lntr_mse, lntr_cos = evaluate_open_loop(model_lntr, val_loader, args.context_steps, args.max_rollout_steps, device)

    print("\n--- Evaluating Control Baseline ---")
    base_mse, base_cos = evaluate_open_loop(model_base, val_loader, args.context_steps, args.max_rollout_steps, device)

    rollout_horizon = len(lntr_mse)
    copy_last_se = torch.zeros(rollout_horizon, device=device)
    total = 0
    for batch in val_loader:
        batch = batch.to(device)
        last_ctx = batch[:, args.context_steps - 1 : args.context_steps, :]
        for step in range(rollout_horizon):
            gt = batch[:, args.context_steps + step : args.context_steps + step + 1, :]
            copy_last_se[step] += F.mse_loss(last_ctx, gt, reduction="none").mean(dim=-1).sum()
        total += batch.shape[0]
    copy_last_mse = (copy_last_se / total).cpu().numpy()

    print("\n" + "=" * 65)
    print(f"{'Horizon':<10} | {'Baseline MSE':<14} | {'LN-TR MSE':<14} | {'Error Reduction'}")
    print("-" * 65)
    for i in range(rollout_horizon):
        reduction = ((base_mse[i] - lntr_mse[i]) / base_mse[i]) * 100
        print(f"Step +{i+1:<4} | {base_mse[i]:<14.4f} | {lntr_mse[i]:<14.4f} | {reduction:+.2f}%")
    print("=" * 65)

    plot_comparison((lntr_mse, lntr_cos), (base_mse, base_cos), copy_last_mse, args.context_steps, output_dir)


if __name__ == "__main__":
    main()