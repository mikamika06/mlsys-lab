import torch
import torch.nn as nn


class ToyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1)
        self.fc = nn.Linear(8 * 4 * 4, 2)

    def forward(self, x):
        raise NotImplementedError


def propagate_pruned_channels(model, pruned_indices):
    raise NotImplementedError
