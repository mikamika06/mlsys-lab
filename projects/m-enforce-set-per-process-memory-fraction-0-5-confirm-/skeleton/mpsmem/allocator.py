def enforce_memory_fraction(fraction: float) -> None:
    """Enforces the MPS per-process memory limit fraction."""
    raise NotImplementedError


def check_allocation_safety(requested_bytes: int, total_device_bytes: int, fraction: float) -> dict:
    """Checks whether an allocation fits within the enforced fraction budget."""
    raise NotImplementedError
