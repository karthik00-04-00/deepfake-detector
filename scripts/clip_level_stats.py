print("clip_level_stats.py loaded")

import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import csv

PRED_FILE = Path("data/processed/frame_preds/frame_predictions.jsonl")
OUT_FILE = Path("data/processed/frame_preds/video_clip_stats.csv")

def main():
    print("Entering main()")
    print("Reading from:", PRED_FILE.resolve())

    data = defaultdict(lambda: defaultdict(list))

    with open(PRED_FILE, "r") as f:
        lines = f.readlines()
        print("Total lines read:", len(lines))

        for line in lines:
            rec = json.loads(line)
            data[rec["video"]][rec["clip"]].append(rec["prob_fake"])

    print("Videos found:", list(data.keys()))

    rows = []

    for video, clips in data.items():
        clip_vars = []
        clip_mean_deltas = []
        clip_max_deltas = []

        for clip, probs in clips.items():
            probs = np.array(probs)
            if len(probs) < 2:
                continue

            deltas = np.abs(np.diff(probs))
            clip_vars.append(probs.var())
            clip_mean_deltas.append(deltas.mean())
            clip_max_deltas.append(deltas.max())

        rows.append({
            "video": video,
            "num_clips": len(clips),
            "mean_clip_var": float(np.mean(clip_vars)),
            "mean_clip_delta": float(np.mean(clip_mean_deltas)),
            "max_clip_delta": float(np.max(clip_max_deltas)),
        })

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video",
                "num_clips",
                "mean_clip_var",
                "mean_clip_delta",
                "max_clip_delta",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print("Saved clip-level stats to", OUT_FILE)

if __name__ == "__main__":
    print("__main__ triggered")
    main()
