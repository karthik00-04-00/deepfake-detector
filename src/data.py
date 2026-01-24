import os
from PIL import Image
from torch.utils.data import Dataset

from src.utils import get_logger

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
logger = get_logger("data")


class DeepfakeDataset(Dataset):
    """
    Dataset for deepfake detection.

    Expected directory structure:
        root_dir/
            train/
                real/
                fake/
            val/
                real/
                fake/
            test/
                real/
                fake/
    """

    def __init__(self, root_dir, split, transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        self.samples = []

        split_dir = os.path.join(root_dir, split)
        real_dir = os.path.join(split_dir, "real")
        fake_dir = os.path.join(split_dir, "fake")

        if not os.path.isdir(real_dir) or not os.path.isdir(fake_dir):
            raise FileNotFoundError(
                f"Expected directories not found in {split_dir}"
            )

        for fname in os.listdir(real_dir):
            if not any(fname.lower().endswith(ext) for ext in IMAGE_EXTS):
                logger.warning("Skipping non-image file: %s", fname)
                continue
            self.samples.append((os.path.join(real_dir, fname), 0))  # real = 0

        for fname in os.listdir(fake_dir):
            if not any(fname.lower().endswith(ext) for ext in IMAGE_EXTS):
                logger.warning("Skipping non-image file: %s", fname)
                continue
            self.samples.append((os.path.join(fake_dir, fname), 1))  # fake = 1

        # 🔴 TEMPORARY: limit dataset size for pipeline sanity check
        # self.samples = self.samples[:200]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label
