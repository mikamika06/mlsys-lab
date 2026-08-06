def max_context_length(
    config: dict,
    dtype: str,
    model_weight_bytes: int,
    non_model_overhead_bytes: int,
    total_vram_bytes: int,
) -> int:
    """Calculates maximum context length that fits in VRAM budget."""
    raise NotImplementedError
