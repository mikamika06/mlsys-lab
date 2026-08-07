import torch
import torch.nn as nn


class ToyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv2(self.relu(self.bn1(self.conv1(x))))


def propagate_channels(pruned_channels):
    groups = {
        "conv1_out": pruned_channels,
        "bn1": pruned_channels,
        "conv2_in": pruned_channels
    }
    return groups
