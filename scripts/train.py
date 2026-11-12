import argparse
import math
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from lntr.data.dataset import LatentSequenceDataset
from lntr.losses.unified_loss import UnifiedLNTRLoss
from lntr.models.forward_predictor import CausalLatentPredictor
from lntr.models.reverse_head import ReverseCycleHead

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (no popup windows)
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 2 & 3: Optimized LN-TR Predictor Training"
    )
    parser.add_argument(
        "--h5_train",
        type=str,
        default="cached_features/vjepa2_vitl_f32_r256/train_latents.h5",
        help="Path to training latents HDF5.",
    )
    parser.add_argument(
        "--h5_val",
        type=str,
        default="cached_features/vjepa2_vitl_f32_r256/val_latents.h5",
        help="Path to validation latents HDF5.",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size (fits in ~2.0 GB VRAM).",
    )
    parser.add_argument(
        "--lr", type=float, default=1.5e-4, help="AdamW base learning rate."
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="AdamW weight decay."
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=512,
        help="Hidden dimension size of predictor.",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=6,
        help="Number of Causal Transformer layers.",
    )
    parser.add_argument(
        "--num_heads", type=int, default=8, help="Number of attention heads."
    )
    parser.add_argument(
        "--lambda_noise",
        type=float,
        default=0.15,
        help="Balanced weight for LSNI noise loss.",
    )
    parser.add_argument(
        "--lambda_cycle",
        type=float,
        default=0.10,
        help="Balanced weight for RLCC cycle loss.",
    )
    parser.add_argument(
        "--k_steps", type=int, default=4, help="Reverse cycle rollout depth K."
    )
    parser.add_argument(
        "--alpha_min",
        type=float,
        default=0.85,
        help="Softer minimum noise coefficient alpha.",
    )
    parser.add_argument(
        "--alpha_max",
        type=float,
        default=1.0,
        help="Maximum noise coefficient alpha.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader multi-processing workers.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints/optimized_lntr",
        help="Directory to save model weights.",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="outputs/tensorboard",
        help="TensorBoard log directory.",
    )
    return parser.parse_args()


def train_epoch(
    model_forward,
    model_reverse,
    dataloader,
    optimizer,
    loss_fn,
    epoch,
    device,
):
    model_forward.train()
    model_reverse.train()

    total_loss = 0.0
    total_mse = 0.0
    total_noisy = 0.0
    total_cycle = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d} [Train]")
    for batch in pbar:
        batch = batch.to(device)
        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, loss_dict = loss_fn(model_forward, model_reverse, batch, epoch)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model_forward.parameters()) + list(model_reverse.parameters()),
            1.0,
        )
        optimizer.step()

        total_loss += loss_dict["l_total"]
        total_mse += loss_dict["l_mse"]
        total_noisy += loss_dict["l_noisy_pred"]
        total_cycle += loss_dict["l_cycle"]

        pbar.set_postfix({
            "loss": f"{loss_dict['l_total']:.4f}",
            "mse": f"{loss_dict['l_mse']:.4f}",
            "noisy": f"{loss_dict['l_noisy_pred']:.4f}",
            "cycle": f"{loss_dict['l_cycle']:.4f}",
            "α": f"{loss_dict['alpha']:.2f}",
        })

    n = len(dataloader)
    return {
        "loss": total_loss / n,
        "mse": total_mse / n,
        "noisy": total_noisy / n,
        "cycle": total_cycle / n,
    }


@torch.no_grad()
def evaluate_and_save_plot(model_forward, model_reverse, dataloader, loss_fn, epoch, device, log_dir):
    model_forward.eval()
    model_reverse.eval()

    total_loss = 0.0
    total_mse = 0.0
    sample_true, sample_pred = None, None

    for batch in dataloader:
        batch = batch.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, loss_dict = loss_fn(model_forward, model_reverse, batch, epoch)

        total_loss += loss_dict["l_total"]
        total_mse += loss_dict["l_mse"]

        if sample_true is None:
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            preds = model_forward(inputs)
            sample_true = targets[0, :, 0].float().cpu().numpy()
            sample_pred = preds[0, :, 0].float().cpu().numpy()

    # --- PLOT 1: Latest Trajectory Comparison (True vs Pred) ---
    plots_dir = Path(log_dir).parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sample_true, label="True Latent ($z_{t+1}$)", color="blue", linewidth=2, marker="o")
    ax.plot(sample_pred, label="Predicted Latent ($\hat{z}_{t+1}$)", color="orange", linestyle="--", linewidth=2, marker="x")
    ax.set_title(f"Trajectory Comparison | Epoch {epoch} | MSE: {total_mse/len(dataloader):.4f}")
    ax.set_xlabel("Time Step (T)")
    ax.set_ylabel("Latent Feature Value (Dim 0)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(plots_dir / "latest_trajectory.png", bbox_inches='tight')
    plt.close(fig)

    n = len(dataloader)
    return {"loss": total_loss / n, "mse": total_mse / n}


def save_training_curves_plot(history, log_dir):
    """--- PLOT 2: Historical Loss & MSE across all epochs ---"""
    plots_dir = Path(log_dir).parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = history["epochs"]
    
    ax.plot(epochs, history["train_loss"], label="Train Total Loss", color="blue", linewidth=2)
    ax.plot(epochs, history["val_loss"], label="Val Total Loss", color="orange", linewidth=2, linestyle="--")
    ax.plot(epochs, history["train_mse"], label="Train MSE", color="green", linewidth=1.5, alpha=0.7)
    ax.plot(epochs, history["val_mse"], label="Val MSE", color="red", linewidth=1.5, linestyle="--", alpha=0.7)
    
    ax.set_title("Training & Validation Metrics Across Epochs (Optimized)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss / MSE Value")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.savefig(plots_dir / "training_curves.png", bbox_inches='tight')
    plt.close(fig)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Starting Optimized LN-TR Model Training on {device} (bfloat16 AMP) ===")

    project_root = Path(__file__).resolve().parent.parent
    h5_train_path = project_root / args.h5_train
    h5_val_path = project_root / args.h5_val
    checkpoint_dir = project_root / args.checkpoint_dir
    log_dir = project_root / args.log_dir

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Datasets & Loaders
    train_dataset = LatentSequenceDataset(h5_train_path)
    val_dataset = LatentSequenceDataset(h5_val_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(
        f"[+] Loaded Datasets: {len(train_dataset)} Train | {len(val_dataset)} Val clips"
    )

    # Models
    model_forward = CausalLatentPredictor(
        input_dim=1024,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    ).to(device)

    model_reverse = ReverseCycleHead(
        input_dim=1024, hidden_dim=args.hidden_dim, num_layers=3
    ).to(device)

    params = list(model_forward.parameters()) + list(model_reverse.parameters())
    total_params = sum(p.numel() for p in params if p.requires_grad)
    print(f"[+] Total Model Parameters: {total_params / 1e6:.2f}M")

    # Loss & Optimizer (Balanced Weights & Softer Noise)
    loss_fn = UnifiedLNTRLoss(
        lambda_noise=args.lambda_noise,
        lambda_cycle=args.lambda_cycle,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        max_epochs=args.epochs,
        k_steps=args.k_steps,
    ).to(device)

    optimizer = torch.optim.AdamW(
        params, lr=args.lr, weight_decay=args.weight_decay
    )

    # 3-Epoch Linear Warmup + Cosine Annealing Scheduler
    def lr_lambda(current_epoch):
        warmup_epochs = 3
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(max(1, warmup_epochs))
        progress = float(current_epoch - warmup_epochs) / float(max(1, args.epochs - warmup_epochs))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    writer = SummaryWriter(log_dir=str(log_dir))

    best_val_loss = float("inf")
    
    # History tracker for the historical curves plot
    history = {
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "train_mse": [],
        "val_mse": []
    }

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model_forward,
            model_reverse,
            train_loader,
            optimizer,
            loss_fn,
            epoch,
            device,
        )
        
        val_metrics = evaluate_and_save_plot(
            model_forward, model_reverse, val_loader, loss_fn, epoch, device, log_dir
        )
        
        scheduler.step()

        # Update history & save historical curves plot
        history["epochs"].append(epoch)
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_mse"].append(train_metrics["mse"])
        history["val_mse"].append(val_metrics["mse"])
        
        save_training_curves_plot(history, log_dir)

        # TensorBoard Logging
        writer.add_scalar("Loss/Train_Total", train_metrics["loss"], epoch)
        writer.add_scalar("Loss/Train_MSE", train_metrics["mse"], epoch)
        writer.add_scalar("Loss/Val_Total", val_metrics["loss"], epoch)
        writer.add_scalar("Loss/Val_MSE", val_metrics["mse"], epoch)
        writer.add_scalar(
            "LNTR/Alpha", loss_fn.lsni_scheduler.get_alpha(epoch), epoch
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss:"
            f" {train_metrics['loss']:.4f} (MSE: {train_metrics['mse']:.4f}) | Val"
            f" Loss: {val_metrics['loss']:.4f} (MSE: {val_metrics['mse']:.4f})"
        )

        # Save Best Checkpoint
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_path = checkpoint_dir / "lntr_full_best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_forward_state_dict": model_forward.state_dict(),
                    "model_reverse_state_dict": model_reverse.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_metrics["loss"],
                },
                best_path,
            )
            print(
                f"  [★] Saved new best checkpoint to {best_path.name} (Val Loss:"
                f" {best_val_loss:.4f})"
            )

    writer.close()
    print(
        f"\n🎉 === Training Complete! Best Val Loss: {best_val_loss:.4f} ==="
    )


if __name__ == "__main__":
    main()