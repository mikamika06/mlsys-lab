import torch
from triton_wrap.kernel import fused_silu_forward, fused_silu_backward


def compute_analytic_grad(x, grad_output):
    return fused_silu_backward(x, grad_output)
