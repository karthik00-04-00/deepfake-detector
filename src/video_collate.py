import torch
from torch.nn.utils.rnn import pad_sequence

def collate_video(batch):
    xs, ys = zip(*batch)

    lengths = torch.tensor([x.shape[0] for x in xs])
    xs_padded = pad_sequence(xs, batch_first=True)  # (B, Lmax, 512)
    ys = torch.stack(ys)

    return xs_padded, lengths, ys
