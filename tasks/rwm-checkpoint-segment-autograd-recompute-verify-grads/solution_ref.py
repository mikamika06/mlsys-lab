import torch


class _Checkpoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, b1, w2, b2):
        ctx.save_for_backward(x, w1, b1, w2, b2)
        with torch.no_grad():
            h = torch.relu(x @ w1.t() + b1)
            y = h @ w2.t() + b2
        ctx.saved_tensor_count = 5
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, w1, b1, w2, b2 = ctx.saved_tensors

        x_r = x.detach().requires_grad_(True)
        w1_r = w1.detach().requires_grad_(True)
        b1_r = b1.detach().requires_grad_(True)
        w2_r = w2.detach().requires_grad_(True)
        b2_r = b2.detach().requires_grad_(True)

        with torch.enable_grad():
            h = torch.relu(x_r @ w1_r.t() + b1_r)
            y = h @ w2_r.t() + b2_r

        grads = torch.autograd.grad(
            y,
            (x_r, w1_r, b1_r, w2_r, b2_r),
            grad_y,
        )
        return grads


def checkpoint_segment(x, w1, b1, w2, b2):
    y = _Checkpoint.apply(x, w1, b1, w2, b2)
    loss = y.sum()
    grads = torch.autograd.grad(
        loss,
        (x, w1, b1, w2, b2),
    )
    return loss.detach(), [g.detach() for g in grads], 5
