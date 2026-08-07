import numpy as np
import torch
import torch.nn as nn


class SampleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(8 * 16 * 16, 10)

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ModelWithCustomOp(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        x = self.fc(x)
        return torch.bitwise_xor(x.int(), 1).float()


def get_sample_inputs():
    torch.manual_seed(42)
    example = (torch.randn(1, 3, 16, 16),)
    eval_inp = (torch.randn(1, 3, 16, 16),)
    return example, eval_inp
