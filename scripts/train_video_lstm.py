import torch
from torch.utils.data import DataLoader
from torch import nn, optim

from src.video_data_video import VideoEmbeddingDataset
from src.video_collate import collate_video
from src.video_model_video import VideoLSTM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    print("train_video_lstm.py loaded")

    dataset = VideoEmbeddingDataset(
        "data/processed/frame_embeddings"
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_video
    )

    model = VideoLSTM().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(10):
        total_loss = 0.0
        for x, lengths, y in loader:
            x = x.to(DEVICE)
            lengths = lengths.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(x, lengths)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}: loss={total_loss/len(loader):.4f}")

if __name__ == "__main__":
    print("__main__ triggered")
    main()
