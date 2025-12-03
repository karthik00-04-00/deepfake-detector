#!/usr/bin/env python3
"""
scripts/count_images.py
Count images in data/raw/ffpp/{real,fake} and write a JSON summary to data/processed/counts.json

Run from project root:
    python scripts/count_images.py
"""
import os
import json
from pathlib import Path

# --- CONFIG (change if your paths differ) ---
PROJECT_ROOT = Path(".").resolve()
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "ffpp"
CLASSES = {"real": RAW_ROOT / "real", "fake": RAW_ROOT / "fake"}
OUT_JSON = PROJECT_ROOT / "data" / "processed" / "counts.json"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

def count_images_in_dir(p: Path) -> int:
    if not p.exists():
        return 0
    count = 0
    for root, _, files in os.walk(p):
        for f in files:
            if Path(f).suffix.lower() in IMAGE_EXTS:
                count += 1
    return count

def find_nonimage_files(p: Path, limit=10):
    """List up to `limit` files with unexpected extensions (help debug)."""
    bad = []
    if not p.exists():
        return bad
    for root, _, files in os.walk(p):
        for f in files:
            if Path(f).suffix.lower() not in IMAGE_EXTS:
                bad.append(str(Path(root) / f))
                if len(bad) >= limit:
                    return bad
    return bad

def main():
    summary = {}
    total = 0
    for cls, p in CLASSES.items():
        n = count_images_in_dir(p)
        summary[cls] = {"path": str(p), "count": n}
        total += n

    summary["total"] = total
    summary["checked_extensions"] = sorted(list(IMAGE_EXTS))

    # Print table
    print("\n=== Image counts ===")
    for cls in CLASSES:
        print(f"  {cls:5} : {summary[cls]['count']:6}  ({summary[cls]['path']})")
    print(f"  {'TOTAL':5} : {summary['total']:6}")
    print("====================\n")

    # Save JSON
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary to: {OUT_JSON}\n")

    # Extra diagnostics
    if summary["total"] == 0:
        print("WARNING: No images found. Check that data/raw/ffpp/real and fake exist and contain image files.")
    else:
        real = summary.get("real", {}).get("count", 0)
        fake = summary.get("fake", {}).get("count", 0)
        if real == 0 or fake == 0:
            print("WARNING: One of the classes has zero images.")
        else:
            ratio = real / fake if fake else float("inf")
            print(f"real:fake = {real}:{fake}  (ratio = {ratio:.3f})")

    # show a few non-image files if present (helps catch mislabeled files)
    for cls, p in CLASSES.items():
        bad = find_nonimage_files(p, limit=5)
        if bad:
            print(f"\nNote: Found {len(bad)} file(s) with non-image extensions under {p} (showing up to 5):")
            for x in bad:
                print("   ", x)

if __name__ == "__main__":
    main()
