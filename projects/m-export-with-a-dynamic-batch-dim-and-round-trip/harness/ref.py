import torch

class DummyModel(torch.nn.Module):
    def forward(self, x):
        return x + 1.0

def get_sample_inputs():
    return (torch.randn(2, 4),)
