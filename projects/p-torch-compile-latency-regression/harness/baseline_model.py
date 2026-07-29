import torch
from torch import nn


class Classifier(nn.Module):
    def __init__(self, d_in=64, d_hidden=128, n_classes=8):
        super().__init__()
        self.proj = nn.Linear(d_in, d_hidden)
        self.mix = nn.Linear(d_hidden, d_hidden)
        self.head = nn.Linear(d_hidden, n_classes)
        self.scale = 1.0

    def forward(self, x):
        h = torch.relu(self.proj(x))
        if h.abs().max().item() > 4.0:
            h = h / 2.0
        h = torch.relu(self.mix(h))
        for i in range(x.shape[0]):
            h[i] = h[i] * self.scale
        return self.head(h)
