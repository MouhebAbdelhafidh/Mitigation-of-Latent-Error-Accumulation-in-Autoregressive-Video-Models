import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from lntr.data.dataset import LatentSequenceDataset, compute_latent_stats
from lntr.losses.unified_loss import UnifiedLNTRLoss
from lntr.models.forward_predictor import CausalLatentPredictor
from lntr.models.reverse_head import ReverseCycleHead


def parse_args():
    p = argparse.ArgumentParser(description="LN-TR predictor training (rollout-aware)")
    p.add_argument("--h5_train", type=str, default="cached_features/vjepa2_vitl_f32_r256/train_latents.h5")
    p.add_argument("--h5_val", type=str, default="cached_features/vjepa2_vitl_f32_r256/val_latents.h5")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--beta2", type=float, default=0.95, help="AdamW beta2 (0.95 is spike-resistant).")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--lambda_noise", type=float, default=0.15)
    p.add_argument("--lambda_cycle", type=float, default=0.10)
    p.add_argument("--k_steps", type=int, default=4, help="RLCC rollout depth.")
    p.add_argument("--alpha_min", type=float, default=0.85)
    p.add_argument("--alpha_max", type=float, default=1.0)
    # Rollout-aware training (off by default)
    p.add_argument("--lambda_roll", type=float, default=0.0, help="Weight of the free-running rollout loss.")
    p.add_argument("--roll_k", type=int, default=8, help="Steps fed back on own predictions.")
    p.add_argument("--roll_ctx_min", type=int, default=2, help="Min ground-truth context for rollout loss.")
    p.add_argument("--roll_detach", type=int, default=1, help="1: detach fed-back predictions (pushforward).")
    # Validation / selection
    p.add_argument("--val_context", type=int, default=4, help="Context steps for validation rollouts.")
    p.add_argument("--select_by", type=str, default="rollout", choices=["rollout", "onestep"],
                   help="Metric used to pick lntr_full_best.pt.")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints/lntr_v3")
    p.add_argument("--log_dir", type=str, default="outputs/tensorboard_v3")
    p.add_argument("--plots_dir", type=str, default="outputs/plots_v3")
    return p.parse_args()


def train_epoch(model_forward, model_reverse, params, dataloader, optimizer,
                scheduler, loss_fn, epoch, device, grad_clip):
    model_forward.train()
    model_reverse.train()

    keys = ("l_total", "l_mse", "l_noisy_pred", "l_cycle", "l_roll")
    sums = {k: 0.0 for k in keys}
    n_ok = n_skipped = 0
    gmax = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d} [Train]")
    for batch in pbar:
        batch = batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            loss, d = loss_fn(model_forward, model_reverse, batch, epoch)

        if not torch.isfinite(loss):
            n_skipped += 1
            continue

        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, grad_clip))
        if not math.isfinite(grad_norm):
            n_skipped += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.step()
        scheduler.step()

        n_ok += 1
        gmax = max(gmax, grad_norm)
        for k in keys:
            sums[k] += d[k]

        pbar.set_postfix({
            "loss": f"{d['l_total']:.3f}",
            "mse": f"{d['l_mse']:.3f}",
            "roll": f"{d['l_roll']:.3f}",
            "gn": f"{grad_norm:.2f}",
        })

    n = max(1, n_ok)
    return {
        "loss": sums["l_total"] / n,
        "mse": sums["l_mse"] / n,
        "noisy": sums["l_noisy_pred"] / n,
        "cycle": sums["l_cycle"] / n,
        "roll": sums["l_roll"] / n,
        "gmax": gmax,
        "skipped": n_skipped,
    }


@torch.no_grad()
def evaluate(model_forward, dataloader, device, val_context):
    """One-step teacher-forced MSE + open-loop rollout MSE (same protocol as baseline_check.py)."""
    model_forward.eval()
    use_amp = device.type == "cuda"
    se = persist_se = 0.0
    count = 0
    roll_se = copy_se = None
    n_clips = 0
    sample = None

    for batch in dataloader:
        batch = batch.to(device, non_blocking=True)
        T = batch.shape[1]
        inputs, targets = batch[:, :-1], batch[:, 1:]

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            preds = model_forward(inputs)
        preds = preds.float()

        se += F.mse_loss(preds, targets, reduction="sum").item()
        persist_se += F.mse_loss(inputs, targets, reduction="sum").item()
        count += targets.numel()

        if sample is None:
            sample = (targets[0, :, 0].cpu().numpy(), preds[0, :, 0].cpu().numpy())

        # Open-loop rollout in fp32 from `val_context` ground-truth steps
        H = T - val_context
        cur = batch[:, :val_context]
        outs = []
        for _ in range(H):
            nxt = model_forward(cur)[:, -1:]
            outs.append(nxt)
            cur = torch.cat([cur, nxt], dim=1)
        roll = torch.cat(outs, dim=1)  # [B, H, D]
        tgt = batch[:, val_context:]
        r = ((roll - tgt) ** 2).mean(dim=2).sum(dim=0)  # [H]
        c = ((batch[:, val_context - 1:val_context] - tgt) ** 2).mean(dim=2).sum(dim=0)
        roll_se = r if roll_se is None else roll_se + r
        copy_se = c if copy_se is None else copy_se + c
        n_clips += batch.shape[0]

    roll_h = (roll_se / n_clips).cpu()
    copy_h = (copy_se / n_clips).cpu()
    late = min(4, len(roll_h))
    metrics = {
        "mse": se / count,
        "persist_mse": persist_se / count,
        "roll_mse": roll_h.mean().item(),
        "roll_late": roll_h[-late:].mean().item(),
        "roll_copy": copy_h.mean().item(),
    }
    return metrics, sample


def save_trajectory_plot(sample, epoch, mse, plots_dir):
    true_traj, pred_traj = sample
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(true_traj, label=r"True latent ($z_{t+1}$)", color="blue", linewidth=2, marker="o")
    ax.plot(pred_traj, label=r"Predicted latent ($\hat{z}_{t+1}$)", color="orange",
            linestyle="--", linewidth=2, marker="x")
    ax.set_title(f"Trajectory Comparison | Epoch {epoch} | Val 1-step MSE: {mse:.4f}")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Standardized latent value (channel 0)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(plots_dir / "latest_trajectory.png", bbox_inches="tight")
    plt.close(fig)


def save_training_curves_plot(history, plots_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    e = history["epochs"]
    ax.plot(e, history["train_loss"], label="Train total loss", color="blue", linewidth=2)
    ax.plot(e, history["train_mse"], label="Train MSE (1-step)", color="green", alpha=0.8)
    ax.plot(e, history["val_mse"], label="Val MSE (1-step)", color="red", linestyle="--", linewidth=2)
    ax.plot(e, history["val_roll"], label="Val rollout MSE (mean)", color="purple", linewidth=2)
    ax.plot(e, history["val_roll_late"], label="Val rollout MSE (last 4 horizons)",
            color="brown", linewidth=2, linestyle="-.")
    ax.axhline(history["roll_copy"][-1], color="gray", linestyle=":", label="Copy-last rollout baseline (val)")
    ax.axhline(history["persist_mse"][-1], color="gray", linestyle="--", alpha=0.5,
               label="Copy-last 1-step baseline (val)")
    ax.set_yscale("log")
    ax.set_title("Training & Validation Metrics (standardized latents, log scale)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss / MSE")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    plt.savefig(plots_dir / "training_curves.png", bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== LN-TR training on {device} (bfloat16 AMP) ===")

    project_root = Path(__file__).resolve().parent.parent
    h5_train_path = project_root / args.h5_train
    h5_val_path = project_root / args.h5_val
    checkpoint_dir = project_root / args.checkpoint_dir
    log_dir = project_root / args.log_dir
    plots_dir = project_root / args.plots_dir
    for d in (checkpoint_dir, log_dir, plots_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("[+] Computing latent statistics on the train split...")
    mean, std = compute_latent_stats(h5_train_path)
    print(f"[+] Channel std: min {std.min():.4f} | median {std.median():.4f} | max {std.max():.4f}")

    train_dataset = LatentSequenceDataset(h5_train_path, mean=mean, std=std)
    val_dataset = LatentSequenceDataset(h5_val_path, mean=mean, std=std)

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    print(f"[+] {len(train_dataset)} train | {len(val_dataset)} val clips")

    model_forward = CausalLatentPredictor(
        input_dim=1024, hidden_dim=args.hidden_dim,
        num_layers=args.num_layers, num_heads=args.num_heads,
    ).to(device)
    model_reverse = ReverseCycleHead(
        input_dim=1024, hidden_dim=args.hidden_dim, num_layers=3
    ).to(device)

    params = list(model_forward.parameters()) + list(model_reverse.parameters())
    print(f"[+] Total parameters: {sum(p.numel() for p in params if p.requires_grad) / 1e6:.2f}M")

    loss_fn = UnifiedLNTRLoss(
        lambda_noise=args.lambda_noise, lambda_cycle=args.lambda_cycle,
        alpha_min=args.alpha_min, alpha_max=args.alpha_max,
        max_epochs=args.epochs, k_steps=args.k_steps,
        lambda_roll=args.lambda_roll, roll_k=args.roll_k,
        roll_ctx_min=args.roll_ctx_min, roll_detach=bool(args.roll_detach),
    ).to(device)

    optimizer = torch.optim.AdamW(
        params, lr=args.lr, betas=(0.9, args.beta2), weight_decay=args.weight_decay
    )

    total_steps = args.epochs * len(train_loader)

    def lr_lambda(step):  # per-iteration linear warmup + cosine decay
        if step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    writer = SummaryWriter(log_dir=str(log_dir))

    best_sel = float("inf")
    best_onestep = float("inf")
    history = {k: [] for k in ("epochs", "train_loss", "train_mse", "val_mse", "persist_mse",
                               "val_roll", "val_roll_late", "roll_copy")}

    def make_ckpt(epoch, vm):
        return {
            "epoch": epoch,
            "model_forward_state_dict": model_forward.state_dict(),
            "model_reverse_state_dict": model_reverse.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_mse": vm["mse"],
            "val_roll_mse": vm["roll_mse"],
            "latent_mean": mean,
            "latent_std": std,
            "args": vars(args),
        }

    for epoch in range(1, args.epochs + 1):
        tm = train_epoch(model_forward, model_reverse, params, train_loader, optimizer,
                         scheduler, loss_fn, epoch, device, args.grad_clip)
        vm, sample = evaluate(model_forward, val_loader, device, args.val_context)

        history["epochs"].append(epoch)
        history["train_loss"].append(tm["loss"])
        history["train_mse"].append(tm["mse"])
        history["val_mse"].append(vm["mse"])
        history["persist_mse"].append(vm["persist_mse"])
        history["val_roll"].append(vm["roll_mse"])
        history["val_roll_late"].append(vm["roll_late"])
        history["roll_copy"].append(vm["roll_copy"])

        save_training_curves_plot(history, plots_dir)
        if sample is not None:
            save_trajectory_plot(sample, epoch, vm["mse"], plots_dir)

        writer.add_scalar("Loss/Train_Total", tm["loss"], epoch)
        writer.add_scalar("Loss/Train_MSE", tm["mse"], epoch)
        writer.add_scalar("Loss/Train_Roll", tm["roll"], epoch)
        writer.add_scalar("Loss/Val_MSE", vm["mse"], epoch)
        writer.add_scalar("Rollout/Val_Mean", vm["roll_mse"], epoch)
        writer.add_scalar("Rollout/Val_Late", vm["roll_late"], epoch)
        writer.add_scalar("Baseline/Val_CopyLast_1step", vm["persist_mse"], epoch)
        writer.add_scalar("Baseline/Val_CopyLast_Rollout", vm["roll_copy"], epoch)
        writer.add_scalar("Opt/GradNorm_Max", tm["gmax"], epoch)
        writer.add_scalar("Opt/LR", optimizer.param_groups[0]["lr"], epoch)

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | Train {tm['loss']:.4f} "
            f"(1-step {tm['mse']:.4f}, noisy {tm['noisy']:.4f}, cycle {tm['cycle']:.4f}, "
            f"roll {tm['roll']:.4f}) | Val 1-step {vm['mse']:.4f} | "
            f"Val rollout {vm['roll_mse']:.4f} (last4 {vm['roll_late']:.4f}, "
            f"copy-last {vm['roll_copy']:.4f}) | max|g| {tm['gmax']:.2f} | skipped {tm['skipped']}"
        )

        torch.save(make_ckpt(epoch, vm), checkpoint_dir / "lntr_full_last.pt")

        sel = vm["roll_mse"] if args.select_by == "rollout" else vm["mse"]
        if sel < best_sel:
            best_sel = sel
            torch.save(make_ckpt(epoch, vm), checkpoint_dir / "lntr_full_best.pt")
            print(f"  [★] New best checkpoint by {args.select_by} ({sel:.4f})")

        # Always keep the best by one-step MSE too, for comparison with earlier runs
        if vm["mse"] < best_onestep:
            best_onestep = vm["mse"]
            torch.save(make_ckpt(epoch, vm), checkpoint_dir / "lntr_best_onestep.pt")

    writer.close()
    print(f"\n=== Training complete. Best by {args.select_by}: {best_sel:.4f} | "
          f"best 1-step val MSE: {best_onestep:.4f} ===")


if __name__ == "__main__":
    main()