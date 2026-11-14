"""Causal V-JEPA 2 latents: each 2-frame tubelet is encoded independently, so latent t
depends only on frames (2t, 2t+1). Nothing from the future can leak into earlier latents
(the full-clip pass in extract_latents.py lets every latent see the whole clip).

Output matches extract_latents.py: one HDF5 dataset per clip id, shape [T/2, 1024].
Run from the project root:
    python scripts/extract_latents_causal.py --num_frames 32 --resolution 256
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

from extract_latents import PROJECT_ROOT, load_video_frames  # same folder (scripts/)

MAX_FAIL_RATE = 0.05


def build_encoder(device):
  out = torch.hub.load("facebookresearch/vjepa2", "vjepa2_vit_large")
  model = out[0] if isinstance(out, (tuple, list)) else out  # (encoder, predictor)
  return model.eval().to(device)


@torch.no_grad()
def encode_tubelets(model, frames, device, chunk_batch):
  """frames [T, 3, H, W] -> [T/2, 1024]; every tubelet is encoded as its own 2-frame clip."""
  T, _, H, W = frames.shape
  P = T // 2
  chunks = frames.reshape(P, 2, 3, H, W).permute(0, 2, 1, 3, 4)  # [P, 3, 2, H, W]
  expected = (H // 16) * (W // 16)
  feats = []
  for i in range(0, P, chunk_batch):
    x = chunks[i:i + chunk_batch].to(device).contiguous()
    with torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
      tokens = model(x)  # [b, N, 1024]
    if tokens.shape[1] != expected:
      raise RuntimeError(f"Unexpected token count {tokens.shape[1]}, expected {expected}")
    feats.append(tokens.float().mean(dim=1))  # spatial mean-pool
  return torch.cat(feats).cpu().numpy().astype(np.float32)


def extract_split(split, config_path, model, device, out_path, args):
  if not config_path.exists():
    print(f"[!] Config not found: {config_path}. Skipping {split}.")
    return 0, 0

  with open(config_path, "r", encoding="utf-8") as f:
    records = json.load(f)
  print(f"\n[+] Split '{split}' ({len(records)} records) -> {out_path.name}")
  out_path.parent.mkdir(parents=True, exist_ok=True)

  reasons = Counter()
  processed = skipped = failed = 0
  with h5py.File(out_path, "a") as h5:
    for rec in tqdm(records, desc=f"Extracting {split}"):
      clip_id = str(rec["id"])
      if clip_id in h5:  # resume support
        skipped += 1
        continue

      vp = rec.get("video_path")
      if not vp:
        failed += 1
        reasons["record has no video_path"] += 1
        continue
      vp = Path(vp)
      if not vp.is_absolute():
        vp = PROJECT_ROOT / vp
      if not vp.exists():
        failed += 1
        reasons["video file not found"] += 1
        continue

      out = load_video_frames(
          vp, num_frames=args.num_frames,
          target_size=(args.resolution, args.resolution),
      )
      frames, err = (out if isinstance(out, tuple) else (out, "decode failed"))
      if frames is None:
        failed += 1
        reasons[err or "decode failed"] += 1
        continue

      latents = encode_tubelets(model, frames, device, args.chunk_batch)
      h5.create_dataset(clip_id, data=latents, compression="lzf")
      processed += 1

  print(f"[{'✓' if failed == 0 else '!'}] '{split}': processed {processed} | cached {skipped} | failed {failed}")
  for reason, count in reasons.most_common(5):
    print(f"    {count:>6} x {reason}")
  return failed, len(records)


def main():
  p = argparse.ArgumentParser(description="Causal (per-tubelet) V-JEPA 2 latent extraction")
  p.add_argument("--num_frames", type=int, default=32, help="Frames per clip (even).")
  p.add_argument("--resolution", type=int, default=256)
  p.add_argument("--chunk_batch", type=int, default=16, help="Tubelets per forward pass.")
  p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
  args = p.parse_args()

  if args.num_frames % 2 != 0:
    raise SystemExit("--num_frames must be even (tubelet size 2).")
  if args.resolution % 16 != 0:
    raise SystemExit("--resolution must be divisible by 16 (patch size 16).")

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"=== Causal V-JEPA 2 extraction on {device} ===")
  model = build_encoder(device)

  cache_dir = (
      PROJECT_ROOT / "cached_features"
      / f"vjepa2_vitl_causal_f{args.num_frames}_r{args.resolution}"
  )
  bad = []
  for split in args.splits:
    failed, total = extract_split(
        split, PROJECT_ROOT / "configs" / f"{split}_subset.json",
        model, device, cache_dir / f"{split}_latents.h5", args,
    )
    if total and failed / total > MAX_FAIL_RATE:
      bad.append(split)

  if bad:
    print(f"\n[!] High failure rate in: {', '.join(bad)}. Check the reasons above.")
  else:
    print("\n=== Done ===")


if __name__ == "__main__":
  main()