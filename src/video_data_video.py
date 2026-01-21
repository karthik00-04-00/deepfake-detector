import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset

class VideoEmbeddingDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        root = Path(root_dir)

        for video_dir in sorted(root.iterdir()):
            if not video_dir.is_dir():
                continue

            label = 1 if "fake" in video_dir.name else 0
            clips = sorted(video_dir.glob("*.npy"))

            if len(clips) == 0:
                continue

            self.samples.append((clips, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        clips, label = self.samples[idx]

        seq = []
        for clip_path in clips:
            clip = np.load(clip_path)      # (T, 512)
            seq.append(clip)

        seq = np.concatenate(seq, axis=0)  # (L, 512)
        x = torch.tensor(seq, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.float32)

        return x, y
