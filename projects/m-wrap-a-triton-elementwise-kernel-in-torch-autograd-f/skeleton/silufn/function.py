import torch
from silufn.kernel import backward_kernel, forward_kernel


class FusedSiluAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError
