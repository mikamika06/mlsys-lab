import torch


def atomic_heavy_backward(x, grad_output):
    return torch.sum(x * grad_output)


def disjoint_write_backward(x, grad_output):
    return torch.sum(x * grad_output)
