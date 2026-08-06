import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 8)

    def forward(self, x):
        return torch.relu(self.fc(x))

def get_test_inputs():
    return (torch.randn(4, 8),)
