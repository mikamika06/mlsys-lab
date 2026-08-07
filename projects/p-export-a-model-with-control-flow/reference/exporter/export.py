import torch
from typing import Dict, Any, Tuple


def get_dynamic_shapes() -> Dict[str, Any]:
    batch_dim = torch.export.Dim("batch", min=1, max=1024)
    seq_dim = torch.export.Dim("seq", min=1, max=2048)
    return {
        "x": {0: batch_dim, 1: seq_dim},
        "cond": {},
        "max_iters": {}
    }


def export_model(model: torch.nn.Module, example_args: Tuple[torch.Tensor, ...]) -> Any:
    dynamic_shapes = get_dynamic_shapes()
    exported = torch.export.export(model, example_args, dynamic_shapes=dynamic_shapes)
    return exported


def run_exported_program(exported_prog: Any, x: torch.Tensor, cond: torch.Tensor, max_iters: torch.Tensor) -> torch.Tensor:
    res = exported_prog.module()(x, cond, max_iters)
    return res
