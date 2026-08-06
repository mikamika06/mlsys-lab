"""Stateful PyTorch decoder conversion utilities."""

from typing import Any, Dict, Tuple


def build_toy_decoder_spec(
    num_layers: int,
    hidden_dim: int,
    num_kv_heads: int,
    head_dim: int,
    max_context: int,
) -> Dict[str, Any]:
    """Build a model specification dictionary for a toy stateful decoder."""
    raise NotImplementedError


def simulate_export(
    spec: Dict[str, Any],
    state_names: Tuple[str, str] = ("key_cache", "value_cache"),
) -> Dict[str, Any]:
    """Simulate exported Core ML spec dictionary containing stateful tensor declarations."""
    raise NotImplementedError
