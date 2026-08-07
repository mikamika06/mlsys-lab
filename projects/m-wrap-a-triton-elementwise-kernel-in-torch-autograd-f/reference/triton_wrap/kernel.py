import torch


def fused_silu_forward(x):
    return x * torch.sigmoid(x)


def fused_silu_backward(x, grad_output):
    sig = torch.sigmoid(x)
    dsig = sig * (1.0 - sig)
    d_silu = sig + x * dsig
    return grad_output * d_silu
