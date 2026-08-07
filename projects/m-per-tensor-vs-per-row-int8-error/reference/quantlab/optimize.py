import torch

def apply_compile_fix(model: torch.nn.Module) -> torch.nn.Module:
    return torch.compile(model)
