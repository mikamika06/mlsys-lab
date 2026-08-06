import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(32, 32)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))

def get_test_inputs():
    return [torch.randn(4, 32), torch.randn(8, 32), torch.randn(4, 32)]
