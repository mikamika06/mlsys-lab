import torch
from triton_autograd.kernels import triton_silu_backward, triton_silu_forward


class TritonSiluFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return triton_silu_forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return triton_silu_backward(x, grad_output)
