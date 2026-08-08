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
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def propagate_pruned_channels(model, pruned_indices):
    return {
        "conv1_out": pruned_indices,
        "bn1_channels": pruned_indices,
        "conv2_in": pruned_indices,
    }
