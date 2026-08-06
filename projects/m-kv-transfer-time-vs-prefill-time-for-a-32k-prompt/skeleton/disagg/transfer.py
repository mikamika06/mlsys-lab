def kv_cache_bytes(prompt_len: int, num_layers: int, num_kv_heads: int, head_dim: int, dtype_bytes: int = 2) -> int:
    """Calculate KV cache memory footprint in bytes."""
    raise NotImplementedError


def prefill_flops(prompt_len: int, model_cfg: dict) -> int:
    """Compute total FLOPs required for prefill stage."""
    raise NotImplementedError


def prefill_time_ms(prompt_len: int, model_cfg: dict, tflops: float) -> float:
    """Calculate prefill compute time in milliseconds."""
    raise NotImplementedError


def transfer_time_ms(kv_bytes: int, bandwidth_gbps: float, latency_ms: float = 0.0) -> float:
    """Calculate KV cache transfer time in milliseconds."""
    raise NotImplementedError


def analyze_kv_transfer(prompt_len: int, model_cfg: dict, hardware_cfg: dict) -> dict:
    """Analyze KV cache bytes, prefill time, transfer time, and their ratio."""
    raise NotImplementedError
