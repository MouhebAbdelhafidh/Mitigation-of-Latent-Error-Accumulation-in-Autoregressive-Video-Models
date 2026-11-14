"""Causal V-JEPA 2 latent extraction for Kinetics-400 text-list format.
Run from project root:
    python scripts/extract_latents_causal_k400.py --num_frames 32 --resolution 256
"""
import argparse
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

from extract_latents_causal import build_encoder, encode_tubelets, MAX_FAIL_RATE
from extract_latents import PROJECT_ROOT, load_video_frames


def parse_val_list(list_path):
    clip_map = {}
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 1:
                filename = parts[0]
                clip_id = Path(filename).stem
                clip_map[clip_id] = filename
    return clip_map


def extract_split_k400(videos_dir, list_path, model, device, out_path, args):
    if not list_path.exists():
        print(f"[!] List file not found: {list_path}.")
        return 0, 0

    clip_map = parse_val_list(list_path)
    print(f"\n[+] Kinetics-400 Validation ({len(clip_map)} records) -> {out_path.name}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    reasons = Counter()
    processed = skipped = failed = 0
    with h5py.File(out_path, "a") as h5:
        for clip_id, filename in tqdm(clip_map.items(), desc="Extracting Kinetics-400"):
            if clip_id in h5:  # resume support
                skipped += 1
                continue

            vp = videos_dir / filename
            if not vp.exists():
                found = list(videos_dir.glob(f"{clip_id}*"))
                if found:
                    vp = found[0]
                else:
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

    print(f"[{'✓' if failed == 0 else '!'}] Processed: {processed} | Cached: {skipped} | Failed: {failed}")
    for reason, count in reasons.most_common(5):
        print(f"     {count:>6} x {reason}")
    return failed, len(clip_map)


def main():
    p = argparse.ArgumentParser(description="Causal V-JEPA 2 latent extraction for Kinetics-400")
    p.add_argument("--num_frames", type=int, default=32, help="Frames per clip (even).")
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--chunk_batch", type=int, default=16, help="Tubelets per forward pass.")
    args = p.parse_args()

    if args.num_frames % 2 != 0:
        raise SystemExit("--num_frames must be even (tubelet size 2).")
    if args.resolution % 16 != 0:
        raise SystemExit("--resolution must be divisible by 16 (patch size 16).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Causal V-JEPA 2 Kinetics-400 extraction on {device} ===")
    model = build_encoder(device)

    videos_dir = PROJECT_ROOT / "data" / "kinetics400" / "videos_val"
    list_path = PROJECT_ROOT / "data" / "kinetics400" / "kinetics400_val_list_videos.txt"
    cache_dir = PROJECT_ROOT / "cached_features" / f"vjepa2_vitl_causal_f{args.num_frames}_r{args.resolution}"
    out_path = cache_dir / "k400_val_latents.h5"

    failed, total = extract_split_k400(videos_dir, list_path, model, device, out_path, args)
    if total and failed / total > MAX_FAIL_RATE:
        print("\n[!] High failure rate detected.")
    else:
        print("\n=== Done ===")


if __name__ == "__main__":
    main()