import torch
from torch.utils.data import DataLoader
from torch import nn, optim

from src.video_data import ClipEmbeddingDataset
from src.video_model import ClipLSTM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    print("train_clip_lstm.py loaded")

    dataset = ClipEmbeddingDataset(
        "data/processed/frame_embeddings"
    )

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True
    )

    model = ClipLSTM().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(5):
        total_loss = 0.0
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}: loss={total_loss/len(loader):.4f}")

if __name__ == "__main__":
    print("__main__ triggered")
    main()
