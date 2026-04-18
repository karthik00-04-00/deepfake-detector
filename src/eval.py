import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torchvision import transforms
import yaml
import argparse

from src.model import get_model
from src.data import DeepfakeDataset


# ---------------------------
# Load YAML config
# ---------------------------
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------
# Transform (match training)
# ---------------------------
def get_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


# ---------------------------
# Evaluation
# ---------------------------
def evaluate(config_path):
    config = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------
    # Load model
    # ---------------------------
    model = get_model()
    model.load_state_dict(
        torch.load(config["eval"]["model_path"], map_location=device)
    )
    model.to(device)
    model.eval()

    # ---------------------------
    # Dataset
    # ---------------------------
    transform = get_transform(config["data"]["image_size"])

    test_dataset = DeepfakeDataset(
    root_dir=config["data"]["data_dir"],  # base dir
    split="test",                         # 🔥 REQUIRED
    transform=transform
)

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=config["data"]["num_workers"]
    )

    # ---------------------------
    # Inference
    # ---------------------------
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = F.softmax(outputs, dim=1)[:, 1]

            preds = (probs > 0.5).long()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # ---------------------------
    # Metrics
    # ---------------------------
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("\n===== Evaluation Results =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")


# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    evaluate(args.config)