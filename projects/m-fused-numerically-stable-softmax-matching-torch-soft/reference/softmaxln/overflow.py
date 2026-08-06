import torch


def naive_softmax_overflow(x):
    e = torch.exp(x)
    return e / torch.sum(e, dim=-1, keepdim=True)
