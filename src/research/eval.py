import os
import json
import yaml
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from src.data import DeepfakeDataset
from src.model import get_model
from src.utils import ensure_dir, get_logger


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_transforms(cfg):
    return transforms.Compose([
        transforms.Resize(
            (cfg["data"]["image_size"], cfg["data"]["image_size"])
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def evaluate(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = get_logger("eval")

    test_tf = get_transforms(cfg)

    test_ds = DeepfakeDataset(
        root_dir=cfg["data"]["data_dir"],
        split="test",
        transform=test_tf
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"]
    )

    # ✅ FIX: correct model construction
    model = get_model(
        num_classes=cfg["model"]["num_classes"],
        pretrained=cfg["model"]["pretrained"]
    ).to(device)

    model_path = os.path.join(
        cfg["logging"]["save_dir"],
        "models",
        "best_baseline.pth"
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "Trained model not found. Run train.py first."
        )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1_score": f1_score(all_labels, all_preds, zero_division=0),
        "auc": roc_auc_score(all_labels, all_probs)
    }

    ensure_dir(os.path.join(cfg["logging"]["save_dir"], "metrics"))

    metrics_path = os.path.join(
        cfg["logging"]["save_dir"],
        "metrics",
        "baseline_metrics.json"
    )

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    for k, v in metrics.items():
        logger.info(f"{k}: {v:.4f}")

    logger.info(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    cfg = load_config("configs/baseline.yaml")
    evaluate(cfg)
