import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

from lntr.models.forward_predictor import CausalLatentPredictor

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_split(h5_path, mean, std):
    with h5py.File(h5_path, "r") as f:
        x = np.stack([f[k][:] for k in f.keys()]).astype(np.float32)  # [N, T, D]
    return (torch.from_numpy(x) - mean) / std


def position_mean(h5_path, mean, std):
    """Mean standardized latent at each time index over train clips -> [T, D]."""
    total, count = None, 0
    with h5py.File(h5_path, "r") as f:
        for k in f.keys():
            x = ((torch.from_numpy(f[k][:]).float() - mean) / std).double()
            total = x if total is None else total + x
            count += 1
    return (total / count).float()


@torch.no_grad()
def rollout(model, ctx, horizon, device, bs=128):
    outs = []
    for i in range(0, ctx.shape[0], bs):
        cur = ctx[i:i + bs].to(device)
        preds = []
        for _ in range(horizon):
            nxt = model(cur)[:, -1:]
            preds.append(nxt)
            cur = torch.cat([cur, nxt], dim=1)
        outs.append(torch.cat(preds, dim=1).cpu())
    return torch.cat(outs)


def load_model(ck, device):
    a = ck["args"]
    sd = dict(ck["model_forward_state_dict"])
    residual = bool(sd["is_residual"].item()) if "is_residual" in sd else False
    sd.setdefault("is_residual", torch.tensor(1 if residual else 0))  # old checkpoints
    model = CausalLatentPredictor(
        1024, a["hidden_dim"], a["num_layers"], a["num_heads"], residual=residual
    )
    model.load_state_dict(sd)
    return model.to(device).eval(), residual


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/lntr_v2/lntr_full_best.pt")
    p.add_argument("--h5_train", default="cached_features/vjepa2_vitl_f32_r256/train_latents.h5")
    p.add_argument("--h5_val", default="cached_features/vjepa2_vitl_f32_r256/val_latents.h5")
    p.add_argument("--context", type=int, default=4)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(PROJECT_ROOT / args.ckpt, map_location="cpu", weights_only=False)
    mean, std = ck["latent_mean"], ck["latent_std"]
    model, residual = load_model(ck, device)
    print(f"checkpoint: {args.ckpt} | {'residual' if residual else 'non-residual'} predictor")

    x = load_split(PROJECT_ROOT / args.h5_val, mean, std)
    pos_mean = position_mean(PROJECT_ROOT / args.h5_train, mean, std)  # [T, D]
    N, T, D = x.shape
    C = args.context
    H = T - C
    print(f"val clips: {N} | T={T} | context={C} | horizon={H}")

    ctx, tgt = x[:, :C], x[:, C:]
    baselines = {
        "model": rollout(model, ctx, H, device),
        "copy_last": ctx[:, -1:].expand(-1, H, -1),
        "ctx_mean": ctx.mean(1, keepdim=True).expand(-1, H, -1),
        "pos_mean": pos_mean[C:].unsqueeze(0).expand(N, -1, -1),  # per-position prior
        "oracle_mean": x.mean(1, keepdim=True).expand(-1, H, -1),  # uses the future!
        "global_mean": torch.zeros_like(tgt),
    }

    names = list(baselines)
    mse = {n: ((b - tgt) ** 2).mean(dim=(0, 2)) for n, b in baselines.items()}
    print("\nhorizon | " + " | ".join(f"{n:>11}" for n in names))
    for h in range(H):
        print(f"{h + 1:>7} | " + " | ".join(f"{mse[n][h].item():>11.4f}" for n in names))
    print("   mean | " + " | ".join(f"{mse[n].mean().item():>11.4f}" for n in names))

    print("\nhorizon | skill vs copy_last (1 - model/copy_last; > 0 means the model beats it)")
    for h in range(H):
        s = 1 - mse["model"][h].item() / mse["copy_last"][h].item()
        print(f"{h + 1:>7} | {s:>+8.3f}")
    s = 1 - mse["model"].mean().item() / mse["copy_last"].mean().item()
    print(f"   mean | {s:>+8.3f}")

    within = ((x - x.mean(1, keepdim=True)) ** 2).mean().item()
    print(f"\nwithin-clip (temporal) variance share: {within:.3f}")


if __name__ == "__main__":
    main()