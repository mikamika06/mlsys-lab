import torch
import torch.nn as nn

class ToyTransformerBlock(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        h = self.norm(x)
        h = torch.relu(self.attn(h))
        return x + self.proj(h)

def get_reference_model():
    return ToyTransformerBlock(64)

def get_reference_inputs():
    return torch.randn(2, 16, 64)
