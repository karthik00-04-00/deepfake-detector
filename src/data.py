import os
from PIL import Image
from torch.utils.data import Dataset


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

        for img in os.listdir(real_dir):
            self.samples.append(
                (os.path.join(real_dir, img), 0)
            )  # real = 0

        for img in os.listdir(fake_dir):
            self.samples.append(
                (os.path.join(fake_dir, img), 1)
            )  # fake = 1

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
