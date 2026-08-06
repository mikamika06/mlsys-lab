import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 16)

    def forward(self, x):
        return torch.sin(self.linear(x))


def generate_inputs():
    torch.manual_seed(42)
    return (torch.randn(4, 16),)


def generate_dynamic_inputs():
    torch.manual_seed(42)
    return [torch.randn(4, 16), torch.randn(8, 16), torch.randn(4, 16)]


def get_reference_metrics():
    return {
        "op_reduction_ratio": 0.4,
        "time_reduction_ratio": 0.3,
        "correctness": 0.0
    }
