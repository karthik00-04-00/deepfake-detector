import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np

def collate_video(batch):
    """
    Collate function for video-level batches.

    batch: list of dicts with keys
      - embeddings: np.ndarray (T, 512)
      - label: int
      - video_id: str
    """

    # Convert embeddings to torch tensors
    embeddings = [
    torch.from_numpy(
        np.ascontiguousarray(item["embeddings"])
    ).float()
    for item in batch
]

    labels = torch.tensor(
        [item["label"] for item in batch],
        dtype=torch.float32
    )

    video_ids = [item["video_id"] for item in batch]

    lengths = torch.tensor([e.shape[0] for e in embeddings])

    embeddings_padded = pad_sequence(
        embeddings,
        batch_first=True
    )  # (B, T_max, 512)

    return {
        "embeddings": embeddings_padded,
        "lengths": lengths,
        "labels": labels,
        "video_id": video_ids
    }
