import torch

def verify_gradients(x_naive, x_fused):
    y_naive = (x_naive * 2.0).sum()
    y_naive.backward()

    y_fused = (x_fused * 2.0).sum()
    y_fused.backward()

    if x_naive.grad is None or x_fused.grad is None:
        return False
    return torch.allclose(x_naive.grad, x_fused.grad, atol=1e-5, rtol=1e-5)
