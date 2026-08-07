import torch


def get_test_tensors():
    torch.manual_seed(42)
    shapes = [(16, 32), (128,), (64, 128)]
    tensors = []
    for shape in shapes:
        x = torch.randn(*shape, dtype=torch.float32)
        grad_out = torch.randn(*shape, dtype=torch.float32)
        tensors.append((x, grad_out))
    return tensors


def torch_silu_forward(x):
    return x * torch.sigmoid(x)


def torch_silu_backward(x, grad_output):
    x_ref = x.detach().clone().requires_grad_(True)
    y_ref = x_ref * torch.sigmoid(x_ref)
    y_ref.backward(grad_output)
    return x_ref.grad
