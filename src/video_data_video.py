import json
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset


class VideoEmbeddingDataset(Dataset):
    """
    Video-level dataset with strict split enforcement.

    One item = one full video
    Temporal modes:
      - "normal"  : original temporal order
      - "shuffle" : random permutation of time
      - "reverse" : reverse temporal order
    """

    def __init__(self, split_file: str, root_dir: str, temporal_mode: str = "normal"):
        self.root_dir = Path(root_dir)
        self.temporal_mode = temporal_mode

        with open(split_file, "r") as f:
            self.video_ids = json.load(f)

        assert len(self.video_ids) > 0, "Split file is empty."

        self.samples = []
        for video_id in self.video_ids:
            video_path = self.root_dir / video_id
            assert video_path.exists(), f"Missing video folder: {video_id}"

            # Label inference
            if video_id.startswith("real_"):
                label = 0
            elif video_id.startswith("fake_"):
                label = 1
            else:
                raise ValueError(f"Cannot infer label from video_id: {video_id}")

            emb_files = sorted(video_path.glob("*.npy"))
            assert len(emb_files) > 0, f"No embeddings found in {video_id}"

            self.samples.append({
                "video_id": video_id,
                "label": label,
                "emb_files": emb_files
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        # Load embeddings: list of (clip_len, 512)
        embeddings = [np.load(p) for p in item["emb_files"]]
        embeddings = np.stack(embeddings)  # (num_clips, clip_len, 512)

        # Flatten clips into a single temporal sequence
        num_clips, clip_len, dim = embeddings.shape
        embeddings = embeddings.reshape(num_clips * clip_len, dim)  # (T, 512)

        T = len(embeddings)

# -------------------------
# Temporal stress tests
# -------------------------
        if self.temporal_mode == "shuffle":
            idx = np.random.permutation(T)
            embeddings = embeddings[idx]

        elif self.temporal_mode == "reverse":
            embeddings = embeddings[::-1]
        elif self.temporal_mode == "first_25":
            embeddings = embeddings[: max(1, int(0.25 * T))]

        elif self.temporal_mode == "first_50":
            embeddings = embeddings[: max(1, int(0.50 * T))]
        elif self.temporal_mode == "last_50":
            embeddings = embeddings[int(0.50 * T):]


        return {
            "video_id": item["video_id"],
            "embeddings": embeddings,
            "label": item["label"]
        }
