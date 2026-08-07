import torch

class SafeCustomOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(x, weight)
        return torch.matmul(x, weight)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        grad_x = torch.matmul(grad_output, weight.t())
        grad_weight = torch.matmul(x.t(), grad_output)
        return grad_x, grad_weight
