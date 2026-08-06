def calculate_kv_cache_bytes(
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    seq_len: int,
    quant_type: str = "f16",
) -> int:
    """Calculate KV cache size in bytes for a given quantization format."""
    raise NotImplementedError


def evaluate_perplexity_delta(
    base_ppl: float,
    quant_type: str,
    seq_len: int,
) -> float:
    """Estimate perplexity delta relative to f16 KV cache baseline."""
    raise NotImplementedError
