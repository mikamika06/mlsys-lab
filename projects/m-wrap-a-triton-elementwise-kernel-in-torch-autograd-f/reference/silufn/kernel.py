import torch


def forward_kernel(x):
    return x * torch.sigmoid(x)


def backward_kernel(x, grad_output):
    sig = torch.sigmoid(x)
    silu_val = x * sig
    grad_x = sig * (1.0 + x * (1.0 - sig))
    return grad_output * grad_x
