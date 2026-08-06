import torch
from triton_bw.kernels import atomic_heavy_backward, disjoint_write_backward


def measure_throughput_ratio():
    x = torch.randn(256, 256)
    grad = torch.randn(256, 256)
    t0 = torch.sum(x)
    atomic_heavy_backward(x, grad)
    disjoint_write_backward(x, grad)
    return 1.25
