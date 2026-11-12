import argparse
import json
from collections import Counter
from pathlib import Path
import h5py
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

try:
  import decord

  decord.bridge.set_bridge("torch")
  HAS_DECORD = True
except ImportError:
  HAS_DECORD = False

try:
  import cv2

  HAS_CV2 = True
except ImportError:
  HAS_CV2 = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAX_FAIL_RATE = 0.05


def parse_args():
  parser = argparse.ArgumentParser(
      description="Phase 1: Extract and cache video latent tensors."
  )
  parser.add_argument(
      "--encoder",
      type=str,
      default="vjepa2_vitl",
      choices=["vjepa2_vitl", "dinov2_vits14"],
      help="Vision encoder to use for feature extraction.",
  )
  parser.add_argument(
      "--num_frames",
      type=int,
      default=32,
      help="Number of frames to sample per clip (must be even for V-JEPA 2).",
  )
  parser.add_argument(
      "--resolution",
      type=int,
      default=224,
      help="Target image spatial resolution (V-JEPA 2 ViT-L was trained at 256).",
  )
  parser.add_argument(
      "--batch_size",
      type=int,
      default=1,
      help="Clips to process concurrently (currently unused).",
  )
  return parser.parse_args()


def _decode_decord(video_path, num_frames):
  vr = decord.VideoReader(str(video_path))
  total = len(vr)
  if total == 0:
    raise RuntimeError("decord: 0 frames")
  indices = np.linspace(0, total - 1, num_frames).astype(int)
  frames = vr.get_batch(indices)  # [T, H, W, C] uint8 (torch bridge)
  return frames.permute(0, 3, 1, 2).float() / 255.0


def _decode_opencv(video_path, num_frames):
  """Sequential read; does not rely on CAP_PROP_FRAME_COUNT (unreliable for webm)."""
  cap = cv2.VideoCapture(str(video_path))
  if not cap.isOpened():
    raise RuntimeError("opencv: cannot open video")
  decoded = []
  while True:
    ret, frame = cap.read()
    if not ret:
      break
    decoded.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
  cap.release()
  if not decoded:
    raise RuntimeError("opencv: decoded 0 frames")
  indices = np.linspace(0, len(decoded) - 1, num_frames).astype(int)
  arr = np.stack([decoded[i] for i in indices])
  return torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0


def load_video_frames(video_path, num_frames=32, target_size=(224, 224)):
  """Returns (frames [T,3,H,W] normalized, None) or (None, error_reason)."""
  errors = []
  frames = None

  if HAS_DECORD:
    try:
      frames = _decode_decord(video_path, num_frames)
    except Exception as e:  # noqa: BLE001
      errors.append(f"decord: {type(e).__name__}: {str(e)[:80]}")

  if frames is None and HAS_CV2:
    try:
      frames = _decode_opencv(video_path, num_frames)
    except Exception as e:  # noqa: BLE001
      errors.append(f"{type(e).__name__}: {str(e)[:80]}")

  if frames is None:
    return None, " | ".join(errors) if errors else "no decoder available"

  transform = transforms.Compose([
      transforms.Resize(target_size, antialias=True),
      transforms.Normalize(
          mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
      ),
  ])
  return transform(frames), None


def build_encoder(encoder_name, device, args):
  """Instantiates the frozen self-supervised vision encoder."""
  print(f"\n[+] Loading encoder: {encoder_name} on {device}...")
  use_amp = device == "cuda"

  if encoder_name == "dinov2_vits14":
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.eval().to(device)
    feature_dim = 384

    def extract_fn(frames_tensor):
      with torch.no_grad():
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=use_amp
        ):
          feats = model(frames_tensor.to(device))
      return feats.float().cpu().numpy().astype(np.float32)

  elif encoder_name == "vjepa2_vitl":
    if args.num_frames % 2 != 0:
      raise ValueError("V-JEPA 2 uses tubelet size 2: --num_frames must be even.")
    if args.resolution % 16 != 0:
      raise ValueError("V-JEPA 2 uses patch size 16: --resolution must be divisible by 16.")

    out = torch.hub.load("facebookresearch/vjepa2", "vjepa2_vit_large")
    model = out[0] if isinstance(out, (tuple, list)) else out  # (encoder, predictor)
    model.eval().to(device)
    feature_dim = 1024
    tubelet, patch = 2, 16

    def extract_fn(frames_tensor):
      T, _, H, W = frames_tensor.shape
      x = frames_tensor.permute(1, 0, 2, 3).unsqueeze(0).to(device)  # [1,3,T,H,W]
      with torch.no_grad():
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=use_amp
        ):
          tokens = model(x)  # [1, N, 1024], tokens ordered (t, h, w)

      Tp, Hp, Wp = T // tubelet, H // patch, W // patch
      if tokens.shape[1] != Tp * Hp * Wp:
        raise RuntimeError(
            f"Unexpected token count {tokens.shape[1]}, expected"
            f" {Tp}x{Hp}x{Wp}={Tp * Hp * Wp}"
        )
      tokens = tokens.float().reshape(1, Tp, Hp * Wp, -1)
      latents = tokens.mean(dim=2)[0]  # spatial mean-pool -> [T/2, 1024]
      return latents.cpu().numpy().astype(np.float32)

  else:
    raise ValueError(f"Unknown encoder: {encoder_name}")

  return extract_fn, feature_dim


def extract_split(split_name, config_path, extract_fn, output_h5_path, args):
  """Extracts features for a split into an HDF5 cache. Returns (processed, cached, failed, total)."""
  if not Path(config_path).exists():
    print(f"[!] Warning: Config file not found at {config_path}. Skipping.")
    return 0, 0, 0, 0

  with open(config_path, "r", encoding="utf-8") as f:
    records = json.load(f)

  print(
      f"\n[+] Processing split '{split_name}' ({len(records)} records) ->"
      f" {output_h5_path.name}"
  )
  output_h5_path.parent.mkdir(parents=True, exist_ok=True)

  reasons = Counter()
  with h5py.File(output_h5_path, "a") as h5_file:
    processed = skipped = failed = 0

    for rec in tqdm(records, desc=f"Extracting {split_name}"):
      clip_id = str(rec["id"])

      if clip_id in h5_file:  # resume support
        skipped += 1
        continue

      video_path = rec.get("video_path")
      if not video_path:
        failed += 1
        reasons["record has no video_path"] += 1
        continue
      video_path = Path(video_path)
      if not video_path.is_absolute():
        video_path = PROJECT_ROOT / video_path
      if not video_path.exists():
        failed += 1
        reasons["video file not found"] += 1
        continue

      frames, err = load_video_frames(
          video_path,
          num_frames=args.num_frames,
          target_size=(args.resolution, args.resolution),
      )
      if frames is None:
        failed += 1
        reasons[err] += 1
        continue

      latents = extract_fn(frames)
      h5_file.create_dataset(clip_id, data=latents, compression="lzf")
      processed += 1

  print(
      f"[{'✓' if failed == 0 else '!'}] Split '{split_name}': Processed:"
      f" {processed} | Already Cached: {skipped} | Failed: {failed}"
  )
  for reason, count in reasons.most_common(5):
    print(f"    {count:>6} x {reason}")
  return processed, skipped, failed, len(records)


def main():
  args = parse_args()
  if not (HAS_DECORD or HAS_CV2):
    raise SystemExit("Install a video decoder: pip install decord opencv-python")
  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"=== Starting Phase 1 Feature Extraction on {device.upper()} ===")
  print(f"Decoders available: decord={HAS_DECORD}, opencv={HAS_CV2}")

  extract_fn, feat_dim = build_encoder(args.encoder, device, args)
  print(f"[✓] Feature Dimension: d = {feat_dim}")

  configs_dir = PROJECT_ROOT / "configs"
  cache_dir = (
      PROJECT_ROOT
      / "cached_features"
      / f"{args.encoder}_f{args.num_frames}_r{args.resolution}"
  )

  splits = [
      ("train", configs_dir / "train_subset.json"),
      ("val", configs_dir / "val_subset.json"),
      ("test", configs_dir / "test_subset.json"),
  ]

  bad_splits = []
  for split_name, config_path in splits:
    h5_path = cache_dir / f"{split_name}_latents.h5"
    _, _, failed, total = extract_split(
        split_name, config_path, extract_fn, h5_path, args
    )
    if total and failed / total > MAX_FAIL_RATE:
      bad_splits.append(split_name)

  if bad_splits:
    print(
        f"\n[!] High failure rate in: {', '.join(bad_splits)}. Check the failure"
        " reasons above before using these caches."
    )
  else:
    print("\n=== All splits extracted and cached ===")


if __name__ == "__main__":
  main()