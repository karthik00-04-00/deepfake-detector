import os
import pandas as pd
import matplotlib.pyplot as plt

from src.inference.video_inference import load_models, predict_video

REAL_DIR = "data/test_videos/real"
FAKE_DIR = "data/test_videos/fake"

load_models(
    backbone_ckpt_path="outputs/models/best_baseline.pth",
    lstm_ckpt_path="outputs/models/video_lstm_last.pth"
)

all_results = []


def evaluate_folder(folder_path, true_label):
    results = []
    correct = 0

    for filename in os.listdir(folder_path):
        if not filename.endswith(".mp4"):
            continue

        path = os.path.join(folder_path, filename)
        result = predict_video(path, mode="normal")
        result_shuffle = predict_video(path, mode="shuffle")
        result_reverse = predict_video(path, mode="reverse")

        predicted = result["label"]
        confidence = result["confidence"]

        is_correct = (predicted == true_label)
        if is_correct:
            correct += 1

        print(
            f"{filename} | "
            f"N: {result['confidence']:.4f} | "
            f"S: {result_shuffle['confidence']:.4f} | "
            f"R: {result_reverse['confidence']:.4f}"
        )

        results.append((filename, predicted, confidence, is_correct))

        all_results.append({
            "video": filename,
            "true_label": true_label,
            "normal_conf": result["confidence"],
            "shuffle_conf": result_shuffle["confidence"],
            "reverse_conf": result_reverse["confidence"],
            "pred_label": predicted,
            "correct": is_correct,
        })

    return correct, len(results)


real_correct, real_total = evaluate_folder(REAL_DIR, "real")
fake_correct, fake_total = evaluate_folder(FAKE_DIR, "fake")

total_correct = real_correct + fake_correct
total = real_total + fake_total

real_acc = real_correct / real_total if real_total > 0 else 0.0
fake_acc = fake_correct / fake_total if fake_total > 0 else 0.0
overall_acc = total_correct / total if total > 0 else 0.0

print("\n--- SUMMARY ---")
print(f"Real Accuracy: {real_correct}/{real_total}")
print(f"Fake Accuracy: {fake_correct}/{fake_total}")
print(f"Overall Accuracy: {total_correct}/{total}")

# Save results to CSV
df = pd.DataFrame(all_results)
df.to_csv("outputs/results/video_evaluation_results.csv", index=False)

# Average confidences across temporal modes
if len(all_results) > 0:
    avg_normal = df["normal_conf"].mean()
    avg_shuffle = df["shuffle_conf"].mean()
    avg_reverse = df["reverse_conf"].mean()
else:
    avg_normal = avg_shuffle = avg_reverse = 0.0

# Accuracy bar plot
plt.figure()
plt.bar(
    ["Real Accuracy", "Fake Accuracy", "Overall Accuracy"],
    [real_acc, fake_acc, overall_acc]
)
plt.ylim(0, 1)
plt.title("Video Classification Accuracy")
plt.savefig("outputs/figures/accuracy_bar.png")
plt.close()

# Temporal comparison plot
plt.figure()
plt.bar(
    ["Normal", "Shuffle", "Reverse"],
    [avg_normal, avg_shuffle, avg_reverse]
)
plt.ylim(0, 1)
plt.title("Average Confidence Across Temporal Modes")
plt.savefig("outputs/figures/temporal_comparison.png")
plt.close()