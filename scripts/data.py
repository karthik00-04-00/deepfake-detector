# src/data.py
"""
Data loader utilities for the deepfake detector.
Assumes processed images are in:
data/processed/ffpp/{train,val,test}/{real,fake}/*.jpg
"""

from pathlib import Path
from typing import Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGE_SIZE = 224

def make_transforms(image_size: int = IMAGE_SIZE):
    train_tf = A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Affine(translate_percent=0.02, scale=(0.95, 1.05), rotate=10, p=0.3),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2()
    ])
    val_tf = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2()
    ])
    return train_tf, val_tf

class ImageFolderDataset(Dataset):
    """
    Simple dataset reading images from folder structure:
    root/{real,fake}/*.jpg
    Returns (image_tensor, label) where label: 0=real, 1=fake
    """
    def __init__(self, root: str or Path, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []
        for cls_name, cls_idx in [("real", 0), ("fake", 1)]:
            folder = self.root / cls_name
            if not folder.exists():
                continue
            for p in sorted(folder.glob("*.jpg")):
                self.samples.append((str(p), cls_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img_np = np.array(img)
        if self.transform:
            augmented = self.transform(image=img_np)
            img_t = augmented["image"]
        else:
            # fallback to simple tensor
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
            img_t = torch.from_numpy(np.array(img).transpose(2,0,1)).float() / 255.0
        return img_t, label

def get_dataloaders(processed_dir: str = "data/processed/ffpp",
                    batch_size: int = 32,
                    num_workers: int = 4,
                    image_size: int = IMAGE_SIZE) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_tf, val_tf = make_transforms(image_size=image_size)
    train_ds = ImageFolderDataset(Path(processed_dir) / "train", transform=train_tf)
    val_ds = ImageFolderDataset(Path(processed_dir) / "val", transform=val_tf)
    test_ds = ImageFolderDataset(Path(processed_dir) / "test", transform=val_tf)

    pin_mem = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_mem)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)
    return train_loader, val_loader, test_loader

# quick sanity helper (can be called manually)
if __name__ == "__main__":
    dl_train, dl_val, dl_test = get_dataloaders(batch_size=8, num_workers=0)
    batch = next(iter(dl_train))
    imgs, labels = batch
    print("batch images shape:", imgs.shape)   # expected: (B, C, H, W)
    print("labels:", labels)
