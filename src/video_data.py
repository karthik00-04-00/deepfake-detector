import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset

class ClipEmbeddingDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        root = Path(root_dir)

        for video_dir in root.iterdir():
            if not video_dir.is_dir():
                continue

            label = 1 if "fake" in video_dir.name else 0

            for clip_path in sorted(video_dir.glob("*.npy")):
                self.samples.append((clip_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        clip_path, label = self.samples[idx]
        x = np.load(clip_path)          # (T, 512)
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.float32)
        return x, y
