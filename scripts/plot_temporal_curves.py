print("plot_temporal_curves.py loaded")

import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

PRED_FILE = Path("data/processed/frame_preds/frame_predictions.jsonl")
OUT_DIR = Path("outputs/plots")

def main():
    print("Entering main()")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    video_probs = defaultdict(list)

    with open(PRED_FILE, "r") as f:
        for line in f:
            rec = json.loads(line)
            video = rec["video"]
            prob = rec["prob_fake"]
            video_probs[video].append(prob)

    print("Videos found:", list(video_probs.keys()))

    for video, probs in video_probs.items():
        plt.figure()
        plt.plot(probs)
        plt.title(f"Temporal fake probability — {video}")
        plt.xlabel("Frame index")
        plt.ylabel("P(fake)")
        plt.ylim(0, 1)

        out_path = OUT_DIR / f"{video}_temporal_curve.png"
        plt.savefig(out_path)
        plt.close()

        print("Saved:", out_path)

if __name__ == "__main__":
    print("__main__ triggered")
    main()
