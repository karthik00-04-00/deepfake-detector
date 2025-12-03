import os
from PIL import Image
from torch.utils.data import Dataset


class DeepfakeDataset(Dataset):
    """
    Minimal dataset for sanity testing.
    Expects structure:
        root/real/*.png|jpg
        root/fake/*.png|jpg
    """

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        self.samples = []

        real_dir = os.path.join(root, "real")
        fake_dir = os.path.join(root, "fake")

        for img in os.listdir(real_dir):
            self.samples.append((os.path.join(real_dir, img), 0))  # real = 0
        for img in os.listdir(fake_dir):
            self.samples.append((os.path.join(fake_dir, img), 1))  # fake = 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label
