import torch
from torch.utils.checkpoint import checkpoint


def run_with_inplace(use_reentrant: bool, x: torch.Tensor):
    def func(inp):
        inp.add_(1.0)
        return inp * 2.0
    try:
        out = checkpoint(func, x, use_reentrant=use_reentrant)
        out.sum().backward()
        return "success"
    except RuntimeError:
        return "runtime_error"
