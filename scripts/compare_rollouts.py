import argparse
from pathlib import Path

import numpy as np
import torch

from baseline_check import load_model, load_split, rollout  # same scripts/ folder

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def per_clip_mse(model, x, context, device):
    H = x.shape[1] - context
    pred = rollout(model, x[:, :context], H, device)
    return ((pred - x[:, context:]) ** 2).mean(dim=2).numpy()  # [N, H]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True,
                   help="label=path/to/ckpt.pt entries; the first one is the reference")
    p.add_argument("--h5_val", default="cached_features/vjepa2_vitl_causal_f32_r256/val_latents.h5")
    p.add_argument("--context", type=int, default=4)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    entries = [item.split("=", 1) for item in args.ckpts]
    labels = [l for l, _ in entries]
    cks = [torch.load(PROJECT_ROOT / path, map_location="cpu", weights_only=False)
           for _, path in entries]

    mean, std = cks[0]["latent_mean"], cks[0]["latent_std"]
    for label, ck in zip(labels, cks):
        if not (torch.allclose(ck["latent_mean"], mean) and torch.allclose(ck["latent_std"], std)):
            print(f"[warn] {label} used different latent statistics than the reference")

    x = load_split(PROJECT_ROOT / args.h5_val, mean, std)
    N, T, D = x.shape
    H = T - args.context
    print(f"val clips: {N} | context={args.context} | horizon={H}")

    results = {}
    for label, ck in zip(labels, cks):
        model, residual = load_model(ck, device)
        results[label] = per_clip_mse(model, x, args.context, device)
        print(f"loaded {label} ({'residual' if residual else 'non-residual'})")
        del model

    ref = labels[0]
    long_start = max(0, H - 4)
    rows = [(f"{h + 1}", slice(h, h + 1)) for h in range(H)]
    rows += [("mean", slice(0, H)), (f"h{long_start + 1}-{H}", slice(long_start, H))]

    print("\nMean rollout MSE")
    print(f"{'horizon':>8} | " + " | ".join(f"{l:>14}" for l in labels))
    for name, sl in rows:
        print(f"{name:>8} | " + " | ".join(
            f"{results[l][:, sl].mean():>14.4f}" for l in labels))

    print(f"\nDifference vs '{ref}' (negative = better than the reference), +/-1.96*SE over val clips")
    print(f"{'horizon':>8} | " + " | ".join(f"{l:>20}" for l in labels[1:]))
    for name, sl in rows:
        cells = []
        for l in labels[1:]:
            d = results[l][:, sl].mean(1) - results[ref][:, sl].mean(1)  # paired per clip
            se = d.std(ddof=1) / np.sqrt(len(d))
            cells.append(f"{d.mean():+.4f}+/-{1.96 * se:.4f}")
        print(f"{name:>8} | " + " | ".join(f"{c:>20}" for c in cells))

    print("\nNote: SE covers val-clip sampling only, not training-seed variance. "
          "Use the difference between two seeds of the same config as the seed-noise estimate.")


if __name__ == "__main__":
    main()