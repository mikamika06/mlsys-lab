"""Reference oracle logic used by test runners."""

from typing import Any, Dict, Tuple

BYTES_PER_ELEMENT = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int8": 1,
}

CONFIGS = [
    {
        "num_layers": 12,
        "num_kv_heads": 8,
        "head_dim": 64,
        "byte_budget": 16 * 1024 * 1024,
        "dtype": "float16",
    },
    {
        "num_layers": 24,
        "num_kv_heads": 16,
        "head_dim": 128,
        "byte_budget": 64 * 1024 * 1024,
        "dtype": "float32",
    },
    {
        "num_layers": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "byte_budget": 128 * 1024 * 1024,
        "dtype": "int8",
    },
]


def compute_kv_bytes(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    context_len: int,
    dtype: str = "float16",
) -> int:
    """Compute exact byte budget needed for storing Key and Value cache tensors."""
    bpe = BYTES_PER_ELEMENT[dtype]
    return 2 * num_layers * num_kv_heads * head_dim * context_len * bpe


def compute_max_context(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    byte_budget: int,
    dtype: str = "float16",
) -> int:
    """Compute maximum context sequence length fitting into the target byte budget."""
    bpe = BYTES_PER_ELEMENT[dtype]
    bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * bpe
    return byte_budget // bytes_per_token


def repair_state_names(
    model_spec: Dict,
    expected_state_names: Tuple[str, str],
) -> Dict:
    """Repair StateType metadata name mismatches in exported Core ML model spec."""
    repaired_spec = {
        "spec_version": model_spec.get("spec_version", 1),
        "description": {
            "states": [],
        },
    }
    raw_states = model_spec.get("description", {}).get("states", [])
    exp_key, exp_val = expected_state_names

    for state in raw_states:
        name = state["name"]
        parts = name.split("_")
        if "key" in name or "k" in parts or parts[-1] != exp_val:
            if "val" in name or "v" in parts or "value" in name:
                new_suffix = exp_val
            else:
                new_suffix = exp_key
        else:
            new_suffix = exp_val

        layer_id = None
        for part in parts:
            if part.isdigit():
                layer_id = part
                break

        if layer_id is not None:
            new_name = f"layer_{layer_id}_{new_suffix}"
        else:
            new_name = name

        new_state = dict(state)
        new_state["name"] = new_name
        repaired_spec["description"]["states"].append(new_state)

    return repaired_spec


def check_state_alignment(
    model_spec: Dict,
    expected_state_names: Tuple[str, str],
) -> bool:
    """Check if model StateType metadata matches the expected state names."""
    states = model_spec.get("description", {}).get("states", [])
    if not states:
        return False
    exp_key, exp_val = expected_state_names
    for state in states:
        name = state.get("name", "")
        if not (name.endswith(f"_{exp_key}") or name.endswith(f"_{exp_val}")):
            return False
    return True
