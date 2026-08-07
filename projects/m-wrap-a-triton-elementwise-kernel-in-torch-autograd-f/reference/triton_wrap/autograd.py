import torch
from triton_wrap.kernel import fused_silu_forward, fused_silu_backward


class FusedSiluAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return fused_silu_forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_x = fused_silu_backward(x, grad_output)
        return grad_x
