import torch
import torch.nn as nn


class CleanModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 16)

    def forward(self, x):
        raise NotImplementedError()
