def calculate_prefix_ram_bytes(model_cfg: dict, active_prefixes: list[dict], retention_hours: float) -> dict:
    """Calculate CPU RAM capacity for prefix KV cache."""
    raise NotImplementedError

def calculate_layer_kv_bytes(layers: int, num_heads: int, head_dim: int, seq_len: int, dtype_bytes: int = 2) -> int:
    """Calculate raw KV cache bytes for a sequence."""
    raise NotImplementedError
