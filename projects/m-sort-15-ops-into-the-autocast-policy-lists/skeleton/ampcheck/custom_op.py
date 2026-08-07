import torch

class SafeCustomOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError
