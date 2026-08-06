import torch
import torch.nn as nn
from typing import Dict, Any


def enable_input_require_grads(model: nn.Module):
    raise NotImplementedError


def run_gradient_flow_experiment(
    model: nn.Module,
    x: torch.Tensor,
    freeze_base: bool,
    use_checkpointing: bool,
    use_input_require_grads: bool
) -> Dict[str, Any]:
    raise NotImplementedError
