import torch
from torch.utils.checkpoint import checkpoint


def count_forward_calls(module, x):
    calls = [0]
    def wrapper(inp):
        calls[0] += 1
        return module(inp)
    out = checkpoint(wrapper, x, use_reentrant=False)
    out.sum().backward()
    return calls[0]
