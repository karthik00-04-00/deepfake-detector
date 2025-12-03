#!/usr/bin/env python3
"""
scripts/split_dataset.py

Deterministic split of data/raw/ffpp/{real,fake} into
data/processed/ffpp/{train,val,test}/{real,fake} by copying files.

Usage (from project root):
    python scripts/split_dataset.py

Config:
  - Change RATIOS or SEED below if needed.
  - By default files are COPIED (raw stays intact). Set COPY=False to move instead.

Outputs:
  - data/processed/ffpp/train/real, fake
  - data/processed/ffpp/val/real, fake
  - data/processed/ffpp/test/real, fake
  - writes a small summary JSON to data/processed/split_summary.json
"""
from pathlib import Path
import random
import shutil
import json
import argparse

# ----------------- CONFIG -----------------
RAW_ROOT = Path("data") / "raw" / "ffpp"
OUT_ROOT = Path("data") / "processed" / "ffpp"
CLASSES = ["real", "fake"]

# Ratios must sum <= 1.0. Remaining goes to test if you choose.
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Deterministic seed so repeated runs produce same split
SEED = 42

# Copy files (True) or move (False)
COPY = True

# Extensions considered images (same as count script)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
# ------------------------------------------

def list_images(folder: Path):
    if not folder.exists():
        return []
    files = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    files_sorted = sorted(files)  # stable sort (by path)
    return files_sorted

def make_dirs():
    for split in ("train", "val", "test"):
        for cls in CLASSES:
            d = OUT_ROOT / split / cls
            d.mkdir(parents=True, exist_ok=True)

def copy_or_move(files, dst_dir: Path):
    for p in files:
        dst = dst_dir / p.name
        if COPY:
            shutil.copy2(p, dst)
        else:
            shutil.move(str(p), str(dst))

def split_list(files, train_ratio, val_ratio):
    n = len(files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    # remaining goes to test
    train = files[:n_train]
    val = files[n_train:n_train + n_val]
    test = files[n_train + n_val:]
    return train, val, test

def main():
    print(f"Starting dataset split from {RAW_ROOT} to {OUT_ROOT}")
    print(f"Config: TRAIN={TRAIN_RATIO}, VAL={VAL_RATIO}, TEST={TEST_RATIO}, SEED={SEED}, COPY={COPY}\n")
    
    random.seed(SEED)

    make_dirs()
    summary = {"seed": SEED, "ratios": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO}, "classes": {}}

    for cls in CLASSES:
        src_dir = RAW_ROOT / cls
        images = list_images(src_dir)
        if not images:
            print(f"WARNING: No images found for class '{cls}' at {src_dir}")
            summary["classes"][cls] = {"found": 0}
            continue

        # Shuffle deterministically by seeding and shuffling a list of indexes
        indices = list(range(len(images)))
        random.shuffle(indices)
        images_shuffled = [images[i] for i in indices]

        train_files, val_files, test_files = split_list(images_shuffled, TRAIN_RATIO, VAL_RATIO)

        copy_or_move(train_files, OUT_ROOT / "train" / cls)
        copy_or_move(val_files,   OUT_ROOT / "val" / cls)
        copy_or_move(test_files,  OUT_ROOT / "test" / cls)

        summary["classes"][cls] = {
            "found": len(images),
            "train": len(train_files),
            "val": len(val_files),
            "test": len(test_files),
        }

        print(f"[{cls}] found={len(images)} -> train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

    # Write summary JSON
    OUT_ROOT.parent.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_ROOT.parent / "split_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print("\nSplit finished. Summary written to:", summary_path)
    print("If you want to reproduce the exact split later, keep SEED the same.")

if __name__ == "__main__":
    main()
