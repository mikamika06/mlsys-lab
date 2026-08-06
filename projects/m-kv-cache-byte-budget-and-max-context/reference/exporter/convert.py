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
    return {
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "max_context": max_context,
        "states": [
            {
                "name": f"layer_{i}_key_cache",
                "shape": [1, num_kv_heads, max_context, head_dim],
                "dtype": "float16",
            }
            for i in range(num_layers)
        ] + [
            {
                "name": f"layer_{i}_value_cache",
                "shape": [1, num_kv_heads, max_context, head_dim],
                "dtype": "float16",
            }
            for i in range(num_layers)
        ],
    }


def simulate_export(
    spec: Dict[str, Any],
    state_names: Tuple[str, str] = ("key_cache", "value_cache"),
) -> Dict[str, Any]:
    """Simulate exported Core ML spec dictionary containing stateful tensor declarations."""
    num_layers = spec["num_layers"]
    key_suffix, val_suffix = state_names
    states = []
    for i in range(num_layers):
        states.append({
            "name": f"layer_{i}_{key_suffix}",
            "shape": [1, spec["num_kv_heads"], spec["max_context"], spec["head_dim"]],
            "dtype": "float16",
        })
        states.append({
            "name": f"layer_{i}_{val_suffix}",
            "shape": [1, spec["num_kv_heads"], spec["max_context"], spec["head_dim"]],
            "dtype": "float16",
        })
    return {
        "spec_version": 1,
        "description": {
            "states": states,
        },
    }
