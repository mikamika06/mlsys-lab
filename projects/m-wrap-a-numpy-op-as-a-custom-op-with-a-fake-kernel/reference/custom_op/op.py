import torch
import numpy as np

lib = torch.library.Library("myops", "DEF")
lib.define("numpy_gelu(Tensor self) -> Tensor")

@torch.library.impl(lib, "numpy_gelu", "cpu")
def numpy_gelu_cpu(x):
    x_np = x.detach().numpy()
    out_np = 0.5 * x_np * (1.0 + np.erf(x_np / np.sqrt(2.0)))
    return torch.from_numpy(out_np).to(x.device)

@torch.library.impl_signature("myops", "numpy_gelu")
def numpy_gelu_sig(x):
    return x.clone()

@torch.library.register_fake("myops::numpy_gelu")
def numpy_gelu_fake(x):
    return torch.empty_like(x)

def numpy_gelu_backward(ctx, grad_output):
    (x,) = ctx.saved_tensors
    x_np = x.detach().numpy()
    grad_np = grad_output.detach().numpy()
    cdf = 0.5 * (1.0 + np.erf(x_np / np.sqrt(2.0)))
    pdf = np.exp(-0.5 * x_np ** 2) / np.sqrt(2.0 * np.pi)
    dx_np = grad_np * (cdf + x_np * pdf)
    return torch.from_numpy(dx_np).to(x.device)

def numpy_gelu_forward(ctx, x):
    ctx.save_for_backward(x)
    return custom_numpy_gelu_forward_kernel(x)

def custom_numpy_gelu_forward_kernel(x):
    return torch.ops.myops.numpy_gelu(x)

torch.library.register_autograd(
    "myops::numpy_gelu",
    numpy_gelu_backward,
    setup_context=numpy_gelu_forward
)

def custom_numpy_gelu(x):
    return torch.ops.myops.numpy_gelu(x)
