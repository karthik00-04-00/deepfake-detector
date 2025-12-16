"""
Minimal wrapper to call the canonical data loader in `src.data`.
This keeps `python scripts/data.py` working for quick checks.
"""

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Ensure repo root is on sys.path so `src` imports work when running script directly
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from src.data import DeepfakeDataset
    from torch.utils.data import DataLoader
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    ds = DeepfakeDataset("data/processed/ffpp/train", transform=transform)
    if len(ds) == 0:
        print("No samples found in data/processed/ffpp/train — make sure dataset is prepared")
    else:
        loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)
        batch = next(iter(loader))
        imgs, labels = batch
        print("batch images shape:", imgs.shape)   # expected: (B, C, H, W)
        print("labels:", labels)
