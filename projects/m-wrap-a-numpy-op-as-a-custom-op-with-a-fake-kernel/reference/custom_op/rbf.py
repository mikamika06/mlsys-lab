import torch
import numpy as np

def numpy_rbf(x: np.ndarray, y: np.ndarray, gamma: float) -> np.ndarray:
    diff = x[:, :, None, :] - y[:, None, :, :]
    dist2 = np.sum(diff ** 2, axis=-1)
    return np.exp(-gamma * dist2)

def numpy_rbf_vjp(grad_out: np.ndarray, x: np.ndarray, y: np.ndarray, gamma: float):
    diff = x[:, :, None, :] - y[:, None, :, :]
    dist2 = np.sum(diff ** 2, axis=-1)
    out = np.exp(-gamma * dist2)
    term = -2.0 * gamma * diff * out[:, :, :, None]
    grad_x = np.sum(grad_out[:, :, :, None] * term, axis=2)
    grad_y = np.sum(grad_out[:, :, :, None] * (-term), axis=1)
    return grad_x, grad_y

@torch.library.custom_op("mylib::rbf", mutates_args=())
def rbf_interact(x: torch.Tensor, y: torch.Tensor, gamma: float) -> torch.Tensor:
    out = numpy_rbf(x.detach().cpu().numpy(), y.detach().cpu().numpy(), gamma)
    return torch.from_numpy(out).to(x.device, x.dtype)

@rbf_interact.register_fake
def _rbf_interact_fake(x: torch.Tensor, y: torch.Tensor, gamma: float) -> torch.Tensor:
    return x.new_empty(x.shape[0], x.shape[1], y.shape[1])

def setup_context(ctx, inputs, output):
    x, y, gamma = inputs
    ctx.save_for_backward(x, y)
    ctx.gamma = gamma

def backward(ctx, grad_output):
    x, y = ctx.saved_tensors
    gamma = ctx.gamma
    gx, gy = numpy_rbf_vjp(grad_output.detach().cpu().numpy(),
                           x.detach().cpu().numpy(),
                           y.detach().cpu().numpy(),
                           gamma)
    return (torch.from_numpy(gx).to(x.device, x.dtype),
            torch.from_numpy(gy).to(y.device, y.dtype),
            None)

rbf_interact.register_autograd(backward, setup_context=setup_context)
