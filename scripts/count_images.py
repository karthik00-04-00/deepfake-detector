#!/usr/bin/env python3
"""
scripts/count_images.py

Counts images in:
data/processed/ffpp/{train,val,test}/{real,fake}

Run from project root:
    python scripts/count_images.py
"""

import os
import json
from pathlib import Path

# --- CONFIG ---
PROJECT_ROOT = Path(".").resolve()
BASE_PATH = PROJECT_ROOT / "data" / "processed" / "ffpp"

SPLITS = ["train", "val", "test"]
CLASSES = ["real", "fake"]

OUT_JSON = PROJECT_ROOT / "data" / "processed" / "split_counts.json"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def count_images(path: Path):
    if not path.exists():
        return 0

    count = 0
    for root, _, files in os.walk(path):
        for f in files:
            if Path(f).suffix.lower() in IMAGE_EXTS:
                count += 1
    return count


def main():
    summary = {}
    total_all = 0

    print("\n=== DATASET SPLIT COUNTS ===")

    for split in SPLITS:
        split_total = 0
        summary[split] = {}

        print(f"\n{split.upper()} SET:")

        for cls in CLASSES:
            path = BASE_PATH / split / cls
            count = count_images(path)

            summary[split][cls] = count
            split_total += count

            print(f"  {cls:5}: {count}")

        summary[split]["total"] = split_total
        total_all += split_total

        print(f"  Total {split}: {split_total}")

    summary["total"] = total_all

    print(f"\nTOTAL IMAGES: {total_all}")
    print("================================\n")

    # Save JSON
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved split summary to: {OUT_JSON}")


if __name__ == "__main__":
    main()