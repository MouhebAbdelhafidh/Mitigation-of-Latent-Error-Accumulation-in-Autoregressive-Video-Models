import argparse
from pathlib import Path

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load(h5_path, max_clips):
  with h5py.File(h5_path, "r") as f:
    keys = list(f.keys())[:max_clips]
    return np.stack([f[k][:] for k in keys]).astype(np.float32)  # [N, T, D]


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--cache_dir", default="cached_features/vjepa2_vitl_f32_r256")
  p.add_argument("--max_clips", type=int, default=3000)
  args = p.parse_args()

  cache = PROJECT_ROOT / args.cache_dir
  tr = load(cache / "train_latents.h5", args.max_clips)
  va = load(cache / "val_latents.h5", args.max_clips)
  print(f"train {tr.shape}  val {va.shape}")
  print(f"non-finite values: train {(~np.isfinite(tr)).sum()}  val {(~np.isfinite(va)).sum()}")

  flat = tr.reshape(-1, tr.shape[-1]).astype(np.float64)
  mu, sd = flat.mean(0), flat.std(0)
  var_ch = sd**2
  print(f"per-channel std: min {sd.min():.4f}  median {np.median(sd):.4f}  max {sd.max():.4f}")
  print(f"top-5 channels hold {100 * np.sort(var_ch)[-5:].sum() / var_ch.sum():.1f}% of total variance")
  print(f"mean latent L2 norm: {np.linalg.norm(flat, axis=1).mean():.2f}")

  mu32 = mu.astype(np.float32)
  x_in, x_tg = va[:, :-1], va[:, 1:]
  print(f"val MSE, predict train mean : {((x_tg - mu32) ** 2).mean():.4f}")
  print(f"val MSE, copy last latent   : {((x_tg - x_in) ** 2).mean():.4f}")

  within = ((tr - tr.mean(axis=1, keepdims=True)) ** 2).mean()
  total = ((tr - mu32) ** 2).mean()
  print(f"within-clip (temporal) share of variance: {100 * within / total:.1f}%")


if __name__ == "__main__":
  main()