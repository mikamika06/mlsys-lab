def check_memory_fit(config: dict, tp: int, pp: int, vram_gb: float) -> bool:
    """Check if model weights, KV cache, and activation memory fit in per-GPU VRAM."""
    raise NotImplementedError


def select_layout(config: dict, vram_gb: float, latency_table: dict) -> int:
    """Find index of valid 8-GPU layout minimizing predicted latency."""
    raise NotImplementedError
