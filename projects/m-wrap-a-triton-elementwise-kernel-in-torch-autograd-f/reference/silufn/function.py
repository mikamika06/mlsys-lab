import torch
from silufn.kernel import backward_kernel, forward_kernel


class FusedSiluAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return forward_kernel(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_x = backward_kernel(x, grad_output)
        return grad_x
