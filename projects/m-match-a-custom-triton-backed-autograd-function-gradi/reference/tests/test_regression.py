import sys
import torch

sys.path.insert(0, ".")
from triton_autograd.func import TritonSiluFunction


def test_silu_autograd_gradient_matches_native_torch():
    torch.manual_seed(123)
    x = torch.randn(32, 64, dtype=torch.float32, requires_grad=True)
    grad_out = torch.randn(32, 64, dtype=torch.float32)

    y_custom = TritonSiluFunction.apply(x)
    y_custom.backward(grad_out)
    custom_grad = x.grad.clone()

    x_ref = x.detach().clone().requires_grad_(True)
    y_ref = x_ref * torch.sigmoid(x_ref)
    y_ref.backward(grad_out)
    ref_grad = x_ref.grad

    rel_err = torch.norm(custom_grad - ref_grad) / (torch.norm(ref_grad) + 1e-8)
    assert rel_err.item() < 1e-4, f"Gradient relative error too high: {rel_err.item()}"


def test_silu_forward_matches_native_torch():
    torch.manual_seed(123)
    x = torch.randn(32, 64, dtype=torch.float32)
    y_custom = TritonSiluFunction.apply(x)
    y_ref = x * torch.sigmoid(x)
    rel_err = torch.norm(y_custom - y_ref) / (torch.norm(y_ref) + 1e-8)
    assert rel_err.item() < 1e-4, f"Forward relative error too high: {rel_err.item()}"
