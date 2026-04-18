import os
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, roc_auc_score,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_DIR = "data/processed/ffpp/test"
MODEL_PATH = "outputs/models/best_baseline.pth"

# -----------------------------
# Load Model
# -----------------------------
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# -----------------------------
# Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Evaluation
# -----------------------------
results = []

for label_name in ["real", "fake"]:
    label = 0 if label_name == "real" else 1
    folder = os.path.join(TEST_DIR, label_name)

    for filename in os.listdir(folder):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        path = os.path.join(folder, filename)

        image = Image.open(path).convert("RGB")
        x = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            fake_conf = probs[0][1].item()

        pred_label = 1 if fake_conf > 0.5 else 0

        results.append({
            "image": filename,
            "true_label": label,
            "pred_label": pred_label,
            "fake_confidence": fake_conf
        })

# -----------------------------
# Save CSV
# -----------------------------
df = pd.DataFrame(results)
os.makedirs("outputs/results", exist_ok=True)
df.to_csv("outputs/results/image_evaluation_results.csv", index=False)

# -----------------------------
# Accuracy
# -----------------------------
accuracy = (df["true_label"] == df["pred_label"]).mean()
print(f"Test Accuracy: {accuracy:.4f}")

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(df["true_label"], df["pred_label"])

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.colorbar()
plt.savefig("outputs/figures/image_confusion_matrix.png")
plt.close()

# -----------------------------
# ROC Curve
# -----------------------------
fpr, tpr, _ = roc_curve(df["true_label"], df["fake_confidence"])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr)
plt.title(f"ROC Curve (AUC = {roc_auc:.3f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.savefig("outputs/figures/image_roc_curve.png")
plt.close()

# -----------------------------
# Confidence Histogram
# -----------------------------
plt.figure()
plt.hist(df[df["true_label"] == 0]["fake_confidence"], bins=30)
plt.hist(df[df["true_label"] == 1]["fake_confidence"], bins=30)
plt.title("Fake Confidence Distribution")
plt.savefig("outputs/figures/image_confidence_distribution.png")
plt.close()

print("Image evaluation complete. Results saved.")
