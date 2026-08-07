import torch
from typing import Dict, Any, Tuple


def get_dynamic_shapes() -> Dict[str, Any]:
    raise NotImplementedError


def export_model(model: torch.nn.Module, example_args: Tuple[torch.Tensor, ...]) -> Any:
    raise NotImplementedError


def run_exported_program(exported_prog: Any, x: torch.Tensor, cond: torch.Tensor, max_iters: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError
