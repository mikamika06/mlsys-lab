import torch
from softmax.kernel import fused_softmax

class FusedSoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        out = fused_softmax(x)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        out, = ctx.saved_tensors
        sum_grad = (grad_output * out).sum(dim=-1, keepdim=True)
        return out * (grad_output - sum_grad)

def fused_softmax_autograd(x):
    return FusedSoftmaxFunction.apply(x)
