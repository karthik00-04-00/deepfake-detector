import torch
import torch.nn as nn

class ClipLSTM(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (B, T, 512)
        _, (h_n, _) = self.lstm(x)
        h = h_n[-1]          # (B, hidden_dim)
        logits = self.fc(h)
        return logits.squeeze(1)
