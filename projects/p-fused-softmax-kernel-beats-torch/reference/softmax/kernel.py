import torch

def fused_softmax(x):
    return torch.softmax(x, dim=-1)
