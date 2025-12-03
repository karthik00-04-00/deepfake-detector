"""
scripts/sanity_run.py

Safe sanity check runner (Windows-friendly).
"""

import multiprocessing


def main():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import transforms

    # Local imports from src/
    from src.model import get_model
    from src.data import DeepfakeDataset
    from src.utils import set_seed, ensure_dir, get_logger

    logger = get_logger("sanity_run")

    # --------------------------------------------------------
    # USER SETTINGS
    # --------------------------------------------------------
    SANITY_ROOT = "data/processed/ffpp/train"   # should contain: real/ and fake/
    NUM_WORKERS = 0        # Windows safe: NO multiprocessing
    BATCH_SIZE = 8
    NUM_EPOCHS = 1         # Keep 1 for very quick test
    LR = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    OUT_CKPT = "outputs/models/sanity_best.pth"
    MAX_BATCHES_PER_EPOCH = 5   # Keep short
    # --------------------------------------------------------

    set_seed(42)
    ensure_dir("outputs/models")

    # transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # Dataset
    ds = DeepfakeDataset(SANITY_ROOT, transform=transform)
    if len(ds) == 0:
        raise RuntimeError(
            f"Dataset empty! No images found under: {SANITY_ROOT}"
        )

    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    # Model
    model = get_model(num_classes=2, pretrained=False).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    logger.info("Starting sanity training...")

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------
    for epoch in range(1, NUM_EPOCHS + 1):
        running_loss = 0
        correct = 0
        total = 0

        for i, (imgs, labels) in enumerate(loader):
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics
            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if i + 1 >= MAX_BATCHES_PER_EPOCH:
                break

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        logger.info(f"Epoch {epoch}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}")

        # Save checkpoint
        torch.save(model.state_dict(), OUT_CKPT)

    print("\nSanity run successful!")
    print("Model saved to:", OUT_CKPT)


# Windows-safe entry point
if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
