import torch
import torch.fx

def suggest_safe_transforms(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """Rewrite simple dynamic allocation and transfer ops into static graph targets."""
    raise NotImplementedError
