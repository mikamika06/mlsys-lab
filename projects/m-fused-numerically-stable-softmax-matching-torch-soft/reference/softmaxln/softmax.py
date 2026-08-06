import torch


def fused_softmax(x):
    m = torch.max(x, dim=-1, keepdim=True).values
    e = torch.exp(x - m)
    return e / torch.sum(e, dim=-1, keepdim=True)
