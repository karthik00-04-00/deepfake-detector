import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score

from src.data import DeepfakeDataset
from src.model import get_model
from src.utils import set_seed, ensure_dir, get_logger


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_transforms(cfg, train=True):
    tfms = [
        transforms.Resize(
            (cfg["data"]["image_size"], cfg["data"]["image_size"])
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]
    return transforms.Compose(tfms)


def train(cfg):
    set_seed(cfg["training"]["seed"])
    logger = get_logger("train")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Transforms
    train_tf = get_transforms(cfg, train=True)
    val_tf = get_transforms(cfg, train=False)

    # Datasets
    train_ds = DeepfakeDataset(
        cfg["data"]["data_dir"],
        "train",
        train_tf
    )
    val_ds = DeepfakeDataset(
        cfg["data"]["data_dir"],
        "val",
        val_tf
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"]
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"]
    )

    # Model
    model = get_model(
        num_classes=cfg["model"]["num_classes"],
        pretrained=cfg["model"]["pretrained"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"]
    )

    ensure_dir(os.path.join(cfg["logging"]["save_dir"], "models"))

    best_auc = 0.0

    for epoch in range(cfg["training"]["epochs"]):
        # -------- TRAIN --------
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            if i == 0:
                logger.info("First batch loaded")
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        # -------- VALIDATION --------
        model.eval()
        all_labels, all_probs = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                probs = torch.softmax(logits, dim=1)[:, 1]

                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        val_auc = roc_auc_score(all_labels, all_probs)

        logger.info(
            f"Epoch [{epoch+1}/{cfg['training']['epochs']}], "
            f"Loss: {avg_loss:.4f}, Val AUC: {val_auc:.4f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(
                model.state_dict(),
                os.path.join(
                    cfg["logging"]["save_dir"],
                    "models",
                    "best_baseline.pth"
                )
            )
            logger.info("Saved new best model")

    logger.info(f"Training complete. Best Val AUC: {best_auc:.4f}")


if __name__ == "__main__":
    cfg = load_config("configs/baseline.yaml")
    train(cfg)
