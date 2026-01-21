import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch import nn, optim

from sklearn.metrics import roc_auc_score

from src.video_data_video import VideoEmbeddingDataset
from src.video_collate import collate_video
from src.video_model_video import VideoLSTM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    print("train_video_lstm.py loaded")
    print("Using device:", DEVICE)
    print("Temporal mode: LAST_50")

    # -------------------------
    # Output directory (DO NOT overwrite other phases)
    # -------------------------
    out_dir = Path("outputs/video_preds_last50")
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Datasets (TIME REVERSED)
    # -------------------------
    train_ds = VideoEmbeddingDataset(
        split_file="splits/video_train.json",
        root_dir="data/processed/frame_embeddings",
        temporal_mode="last_50"
    )

    val_ds = VideoEmbeddingDataset(
        split_file="splits/video_val.json",
        root_dir="data/processed/frame_embeddings",
        temporal_mode="last_50"
    )
    print(f"Train videos: {len(train_ds)}")
    print(f"Val videos:   {len(val_ds)}")

    # -------------------------
    # DataLoaders
    # -------------------------
    train_loader = DataLoader(
        train_ds,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_video
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_video
    )

    # -------------------------
    # Model
    # -------------------------
    model = VideoLSTM().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # -------------------------
    # Training + Validation
    # -------------------------
    for epoch in range(10):

        # ---- Train ----
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            x = batch["embeddings"].to(DEVICE)
            lengths = batch["lengths"].to(DEVICE)
            y = batch["labels"].float().to(DEVICE)

            optimizer.zero_grad()

            logits = model(x, lengths)
            if logits.dim() == 2:
                logits = logits.squeeze(1)

            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        val_records = []

        with torch.no_grad():
            for batch in val_loader:
                x = batch["embeddings"].to(DEVICE)
                lengths = batch["lengths"].to(DEVICE)
                y = batch["labels"].float().to(DEVICE)
                video_ids = batch["video_id"]

                logits = model(x, lengths)
                if logits.dim() == 2:
                    logits = logits.squeeze(1)

                loss = criterion(logits, y)
                val_loss += loss.item()

                probs = torch.sigmoid(logits)

                for vid, label, prob in zip(video_ids, y, probs):
                    val_records.append({
                        "video_id": vid,
                        "label": int(label.item()),
                        "prob": float(prob.item())
                    })

        val_loss /= len(val_loader)

        # ---- Save predictions ----
        with open(out_dir / f"epoch_{epoch:02d}.json", "w") as f:
            json.dump(val_records, f, indent=2)

        # ---- Metrics ----
        labels = [r["label"] for r in val_records]
        probs = [r["prob"] for r in val_records]

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}",
            end=""
        )

        if len(set(labels)) > 1:
            auc = roc_auc_score(labels, probs)
            print(f" | Val AUC: {auc:.4f}")
        else:
            print(" | Val AUC: undefined (single class)")


if __name__ == "__main__":
    main()
