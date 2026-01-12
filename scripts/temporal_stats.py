import json
from pathlib import Path
import numpy as np
import csv
from collections import defaultdict

PRED_FILE = Path("data/processed/frame_preds/frame_predictions.jsonl")
OUT_FILE = Path("data/processed/frame_preds/video_stats.csv")

def main():
    print("Starting temporal stats computation...")
    print("Reading from:", PRED_FILE.resolve())

    video_probs = defaultdict(list)

    with open(PRED_FILE, "r") as f:
        lines = f.readlines()
        print("Total lines read:", len(lines))

        for line in lines:
            rec = json.loads(line)
            video = rec["video"]
            prob = rec["prob_fake"]
            video_probs[video].append(prob)

    rows = []
    for video, probs in video_probs.items():
        probs = np.array(probs)
        rows.append({
            "video": video,
            "num_frames": len(probs),
            "mean_prob": float(probs.mean()),
            "var_prob": float(probs.var()),
            "max_prob": float(probs.max()),
        })

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["video", "num_frames", "mean_prob", "var_prob", "max_prob"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Saved video-level stats to {OUT_FILE}")

if __name__ == "__main__":
    main()
