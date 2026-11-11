"""
Build train/val/test subsets of Something-Something v2 from local files.

Outputs (under PROJECT_ROOT):
    data/train_clips/*.webm   configs/train_subset.json   (10,000 clips)
    data/val_clips/*.webm     configs/val_subset.json     ( 2,000 clips)
    data/test_clips/*.webm    configs/test_subset.json    ( 2,000 clips)

"""

import csv
import json
import random
import shutil
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent

SEED = 42
SHUFFLE = True  # True: seeded random sample; False: first N available in file order

OUT_DATA = PROJECT_ROOT / "data"
OUT_CONFIGS = PROJECT_ROOT / "configs"

SPLITS = [
    # name, label file, number of clips, output clip dir, output config
    ("train", "train.json", 10_000, OUT_DATA / "train_clips", OUT_CONFIGS / "train_subset.json"),
    ("val", "validation.json", 2_000, OUT_DATA / "val_clips", OUT_CONFIGS / "val_subset.json"),
    ("test", "test.json", 2_000, OUT_DATA / "test_clips", OUT_CONFIGS / "test_subset.json"),
]


# --------------------------------------------------------------------------- #
# Path discovery
# --------------------------------------------------------------------------- #
def has_webm(d: Path) -> bool:
    return d.is_dir() and next(d.glob("*.webm"), None) is not None


def find_video_dir() -> Path | None:
    root_clips = PROJECT_ROOT / "clips"
    candidates = [
        root_clips,
        root_clips / "20bn-something-something-v2",
        root_clips / "20bn-something-something-v2" / "20bn-something-something-v2",
        PROJECT_ROOT / "20bn-something-something-v2",
        PROJECT_ROOT / "20bn-something-something-v2" / "20bn-something-something-v2",
    ]
    for c in candidates:
        if has_webm(c):
            return c
    # Fallback: search anywhere under clips/ for the first .webm and use its folder.
    if root_clips.is_dir():
        first = next(root_clips.rglob("*.webm"), None)
        if first is not None:
            return first.parent
    return None


def find_labels_dir() -> Path | None:
    for c in (PROJECT_ROOT / "labels", PROJECT_ROOT / "labels" / "labels"):
        if (c / "train.json").is_file():
            return c
    base = PROJECT_ROOT / "labels"
    if base.is_dir():
        first = next(base.rglob("train.json"), None)
        if first is not None:
            return first.parent
    return None


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #
def load_json_entries(path: Path) -> list[dict]:
    """Load a label file that is either a JSON array or JSON Lines."""
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
        if isinstance(data, dict):
            if "id" in data:
                return [data]
            return [{"id": k, "label": v} for k, v in data.items()]
    except json.JSONDecodeError:
        pass

    entries = []
    for lineno, line in enumerate(text.splitlines(), 1):
        line = line.strip().rstrip(",")
        if not line or line in ("[", "]"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            print(f"  [warn] {path.name}: skipping unparsable line {lineno}")
            continue
        if isinstance(obj, dict):
            entries.append(obj)
    return entries


def _clean_id(value) -> str:
    """Normalise ids coming from CSV/Excel (e.g. 1420.0 -> '1420')."""
    s = str(value).strip()
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return s


def load_test_answers(labels_dir: Path) -> dict[str, str]:
    """Load test-answers (.csv / .xlsx / .xls) -> {id: label}."""
    matches = sorted(labels_dir.glob("test-answers*"))
    if not matches:
        print("  [info] test-answers file not found; test labels will be omitted.")
        return {}
    path = matches[0]
    answers: dict[str, str] = {}

    if path.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        try:
            import pandas as pd
            df = pd.read_excel(path, header=None, dtype=str)
        except Exception as err:  # noqa: BLE001
            print(f"  [warn] Could not read {path.name} ({err}). "
                  f"Try: pip install openpyxl pandas. Test labels omitted.")
            return {}
        for _, row in df.iterrows():
            vals = [v for v in row.tolist() if isinstance(v, str) and v.strip()]
            if len(vals) < 2:
                continue
            vid = _clean_id(vals[0])
            if vid.lower() == "id":
                continue
            answers[vid] = vals[1].strip()
    else:
        raw = path.read_text(encoding="utf-8-sig")
        try:
            delimiter = csv.Sniffer().sniff(raw[:4096], delimiters=";,\t|").delimiter
        except csv.Error:
            delimiter = ";"
        for row in csv.reader(raw.splitlines(), delimiter=delimiter):
            if len(row) < 2:
                continue
            vid = _clean_id(row[0])
            if vid.lower() == "id":
                continue
            answers[vid] = ",".join(row[1:]).strip() if len(row) > 2 else row[1].strip()

    print(f"  [info] Loaded {len(answers)} test answers from {path.name}")
    return answers


# --------------------------------------------------------------------------- #
# Core
# --------------------------------------------------------------------------- #
def build_split(name, label_file, n_clips, clip_dir, config_path,
                video_dir: Path, labels_dir: Path, answers=None):
    print(f"\n=== {name.upper()} ===")
    label_path = labels_dir / label_file
    if not label_path.is_file():
        print(f"  [error] Label file not found: {label_path}")
        return 0

    entries = load_json_entries(label_path)
    print(f"  Parsed {len(entries)} entries from {label_file}")

    available = []
    seen = set()
    missing = 0
    for e in entries:
        if "id" not in e:
            continue
        vid = _clean_id(e["id"])
        if not vid or vid in seen:
            continue
        seen.add(vid)
        if (video_dir / f"{vid}.webm").is_file():
            e = dict(e)
            e["id"] = vid
            available.append(e)
        else:
            missing += 1
    print(f"  {len(available)} have a matching .webm on disk ({missing} missing)")

    if not available:
        print("  [error] No matching videos.")
        print(f"          video_dir: {video_dir}")
        print(f"          sample files on disk: {[p.name for p in list(video_dir.glob('*.webm'))[:5]]}")
        print(f"          sample ids in JSON:   {[str(x.get('id')) for x in entries[:5]]}")
        return 0

    if len(available) < n_clips:
        print(f"  [warn] Only {len(available)} available; requested {n_clips}.")

    if SHUFFLE:
        random.Random(SEED).shuffle(available)
    selected = available[:n_clips]

    clip_dir.mkdir(parents=True, exist_ok=True)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    subset = []
    copied = 0
    for i, e in enumerate(selected, 1):
        vid = e["id"]
        src = video_dir / f"{vid}.webm"
        dst = clip_dir / f"{vid}.webm"
        try:
            if not (dst.is_file() and dst.stat().st_size == src.stat().st_size):
                shutil.copy2(src, dst)
            copied += 1
        except OSError as err:
            print(f"  [warn] failed to copy {src.name}: {err}")
            continue

        record = dict(e)
        if answers is not None:
            record["label"] = answers.get(vid, record.get("label"))
        record["video_path"] = str(dst.relative_to(PROJECT_ROOT)).replace("\\", "/")
        subset.append(record)

        if i % 1000 == 0:
            print(f"  ... {i}/{len(selected)} copied")

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(subset, f, indent=2, ensure_ascii=False)

    print(f"  Copied {copied} clips -> {clip_dir}")
    print(f"  Saved {len(subset)} entries -> {config_path}")
    return copied


def main():
    video_dir = find_video_dir()
    labels_dir = find_labels_dir()

    print(f"PROJECT_ROOT : {PROJECT_ROOT}")
    print(f"video_dir    : {video_dir}")
    print(f"labels_dir   : {labels_dir}")
    if video_dir is None:
        sys.exit("No .webm files found under <PROJECT_ROOT>\\clips. "
                 "Make sure this script sits in the project root.")
    if labels_dir is None:
        sys.exit("train.json not found under <PROJECT_ROOT>\\labels.")

    test_answers = load_test_answers(labels_dir)

    totals = {}
    for name, label_file, n, clip_dir, cfg in SPLITS:
        answers = test_answers if name == "test" else None
        totals[name] = build_split(name, label_file, n, clip_dir, cfg,
                                   video_dir, labels_dir, answers)

    print("\n=== SUMMARY ===")
    for (name, _, n, _, _) in SPLITS:
        status = "OK" if totals[name] == n else "SHORT"
        print(f"  {name:<5} {totals[name]:>6} / {n:<6} [{status}]")


if __name__ == "__main__":
    main()