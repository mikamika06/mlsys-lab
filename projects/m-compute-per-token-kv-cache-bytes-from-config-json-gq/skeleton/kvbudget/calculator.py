def bytes_per_element(dtype: str) -> int:
    """Return byte size for a given dtype string."""
    raise NotImplementedError


def compute_per_token_kv_bytes(config: dict, kv_cache_dtype: str = "auto") -> int:
    """Compute KV cache bytes per token across all layers accounting for GQA."""
    raise NotImplementedError
