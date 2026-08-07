import torch
import numpy as np

@torch.library.custom_op("customop::numpy_scale_op", mutates_args=())
def custom_op_func(x: torch.Tensor, scale: float) -> torch.Tensor:
    x_np = x.detach().cpu().numpy()
    out_np = x_np * np.sin(x_np) * scale
    return torch.tensor(out_np, dtype=x.dtype, device=x.device)

@custom_op_func.register_fake
def _(x: torch.Tensor, scale: float) -> torch.Tensor:
    return torch.empty_like(x)

def _backward(ctx, grad_output):
    x, = ctx.saved_tensors
    scale, = ctx.saved_data
    x_np = x.detach().cpu().numpy()
    grad_np = grad_output.detach().cpu().numpy()
    dx_np = grad_np * scale * (np.sin(x_np) + x_np * np.cos(x_np))
    return torch.tensor(dx_np, dtype=x.dtype, device=x.device), None

@custom_op_func.register_autograd
def _(ctx, x, scale):
    ctx.save_for_backward(x)
    ctx.saved_data = (scale,)
    return custom_op_func(x, scale)

custom_op_func.register_autograd(backward=_backward)

def find_bad_dim(shape_list, expected_shape) -> int:
    for i, s in enumerate(shape_list):
        if s != expected_shape[i]:
            return i
    return -1
