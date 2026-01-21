import torch
import torch.nn as nn

class VideoLSTM(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        h = h_n[-1]                 # (B, hidden)
        logits = self.fc(h)
        return logits.squeeze(1)
