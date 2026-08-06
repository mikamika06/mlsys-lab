def compute_recommended_wired_limit(hw_memsize_bytes: int) -> int:
    """Compute recommended set_wired_limit in bytes for a given hw.memsize."""
    raise NotImplementedError


def compute_recommended_cache_limit(hw_memsize_bytes: int, active_model_bytes: int) -> int:
    """Compute recommended set_cache_limit in bytes given model size and memory limit."""
    raise NotImplementedError
