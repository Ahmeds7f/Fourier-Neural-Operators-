import torch
import torch.nn as nn
from neuralop.models.fno import FNO

# the values in the CNN and FNO are initiated such that both models have ~470,000 parameters
class CNN(nn.Module):
    def __init__(self, hidden_channels=80, kernel=3, n_layers=8):
        super().__init__()
        padding = (kernel - 1) // 2

        self.inc = nn.Conv2d(2, hidden_channels, kernel, padding=padding)

        self.convs = nn.ModuleList([
            nn.Conv2d(hidden_channels, hidden_channels, kernel, padding=padding)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([
            nn.GroupNorm(1, hidden_channels)
            for _ in range(n_layers)
        ])


        self.outc = nn.Conv2d(hidden_channels, 2, kernel, padding=padding)
        self.relu = nn.ReLU()

    def forward(self, x, **kwargs):
        x = self.relu(self.inc(x))

        for conv, norm in zip(self.convs, self.norms):
            identity = x
            out = norm(conv(x))
            x = self.relu(out + identity)

        return self.outc(x)


def get_fno(n_modes=16, hidden_channels=20, n_layers=4):
    return FNO(n_modes=(n_modes, n_modes),
               hidden_channels=hidden_channels,
               in_channels=2, out_channels=2,
               n_layers=n_layers, use_channel_mlp=True)
