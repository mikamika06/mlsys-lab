import torch
import torch.fx

def check_graph_violations(gm: torch.fx.GraphModule) -> list[dict]:
    """Detect CUDA Graph capture violations in FX GraphModule."""
    raise NotImplementedError
