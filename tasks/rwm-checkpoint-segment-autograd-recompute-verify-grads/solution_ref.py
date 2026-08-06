import torch


class _Checkpoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, b1, w2, b2):
        ctx.save_for_backward(x, w1, b1, w2, b2)
        with torch.no_grad():
            h_shape_0 = x.shape[0]
            h_shape_1 = w1.shape[0]
            h = x.new_zeros((h_shape_0, h_shape_1))
            for i in range(h_shape_0):
                for j in range(h_shape_1):
                    acc = 0.0
                    for k in range(x.shape[1]):
                        acc = acc + x[i, k] * w1[j, k]
                    acc = acc + b1[j]
                    if acc < 0.0:
                        h[i, j] = 0.0
                    else:
                        h[i, j] = acc

            y_shape_0 = h.shape[0]
            y_shape_1 = w2.shape[0]
            y = h.new_zeros((y_shape_0, y_shape_1))
            for i in range(y_shape_0):
                for j in range(y_shape_1):
                    acc = 0.0
                    for k in range(h.shape[1]):
                        acc = acc + h[i, k] * w2[j, k]
                    acc = acc + b2[j]
                    y[i, j] = acc
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
            h_shape_0 = x_r.shape[0]
            h_shape_1 = w1_r.shape[0]
            h = x_r.new_zeros((h_shape_0, h_shape_1))
            for i in range(h_shape_0):
                for j in range(h_shape_1):
                    acc = 0.0
                    for k in range(x_r.shape[1]):
                        acc = acc + x_r[i, k] * w1_r[j, k]
                    acc = acc + b1_r[j]
                    if acc < 0.0:
                        h[i, j] = 0.0
                    else:
                        h[i, j] = acc

            y_shape_0 = h.shape[0]
            y_shape_1 = w2_r.shape[0]
            y = h.new_zeros((y_shape_0, y_shape_1))
            for i in range(y_shape_0):
                for j in range(y_shape_1):
                    acc = 0.0
                    for k in range(h.shape[1]):
                        acc = acc + h[i, k] * w2_r[j, k]
                    acc = acc + b2_r[j]
                    y[i, j] = acc

        grads = torch.autograd.grad(
            y,
            (x_r, w1_r, b1_r, w2_r, b2_r),
            grad_y,
        )
        return grads


def checkpoint_segment(x, w1, b1, w2, b2):
    y = _Checkpoint.apply(x, w1, b1, w2, b2)
    loss_acc = 0.0
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            loss_acc = loss_acc + y[i, j]
    loss = loss_acc
    grads = torch.autograd.grad(
        loss,
        (x, w1, b1, w2, b2),
    )
    return loss.detach(), [g.detach() for g in grads], 5
