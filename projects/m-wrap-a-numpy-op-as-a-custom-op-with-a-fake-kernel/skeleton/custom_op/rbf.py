import torch
import numpy as np

def numpy_rbf(x: np.ndarray, y: np.ndarray, gamma: float) -> np.ndarray:
    raise NotImplementedError

def numpy_rbf_vjp(grad_out: np.ndarray, x: np.ndarray, y: np.ndarray, gamma: float):
    raise NotImplementedError

def rbf_interact(x: torch.Tensor, y: torch.Tensor, gamma: float) -> torch.Tensor:
    raise NotImplementedError

def _rbf_interact_fake(x: torch.Tensor, y: torch.Tensor, gamma: float) -> torch.Tensor:
    raise NotImplementedError

def setup_context(ctx, inputs, output):
    raise NotImplementedError

def backward(ctx, grad_output):
    raise NotImplementedError
