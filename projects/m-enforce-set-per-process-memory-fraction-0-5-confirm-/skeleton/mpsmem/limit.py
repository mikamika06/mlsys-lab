def configure_memory_fraction(fraction: float) -> None:
    """Configures the MPS per-process memory limit fraction."""
    raise NotImplementedError


def check_oom_threshold(limit_fraction: float, total_budget_bytes: int) -> dict:
    """Simulates or triggers allocations to confirm early OOM under fraction limits."""
    raise NotImplementedError
