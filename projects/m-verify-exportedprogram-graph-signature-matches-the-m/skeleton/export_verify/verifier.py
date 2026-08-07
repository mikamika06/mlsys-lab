import tempfile
from typing import Any, Dict, Tuple
import torch
import torch.export


def verify_graph_signature(
    mod: torch.nn.Module,
    ep: torch.export.ExportedProgram,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any] | None = None,
) -> Tuple[bool, Dict[str, Any]]:
    raise NotImplementedError


def verify_roundtrip_equivalence(
    ep: torch.export.ExportedProgram,
    sample_args: Tuple[Any, ...],
    sample_kwargs: Dict[str, Any] | None = None,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> bool:
    raise NotImplementedError


def inspect_strict_export_behavior(
    mod: torch.nn.Module,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    raise NotImplementedError
