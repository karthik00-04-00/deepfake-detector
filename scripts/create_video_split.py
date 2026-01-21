import json
import random
from pathlib import Path

random.seed(42)

DATA_ROOT = Path("data/processed/frame_embeddings")
OUT_DIR = Path("splits")
OUT_DIR.mkdir(exist_ok=True)

VAL_RATIO = 0.2

def main():
    videos = []

    for video_dir in DATA_ROOT.iterdir():
        if video_dir.is_dir():
            name = video_dir.name
            if name.startswith("real_") or name.startswith("fake_"):
                videos.append(name)

    assert len(videos) > 0, "No video folders found."

    random.shuffle(videos)

    n_val = max(1, int(len(videos) * VAL_RATIO))
    val_videos = videos[:n_val]
    train_videos = videos[n_val:]

    with open(OUT_DIR / "video_train.json", "w") as f:
        json.dump(train_videos, f, indent=2)

    with open(OUT_DIR / "video_val.json", "w") as f:
        json.dump(val_videos, f, indent=2)

    print(f"Train videos: {len(train_videos)}")
    print(f"Val videos:   {len(val_videos)}")

if __name__ == "__main__":
    main()
