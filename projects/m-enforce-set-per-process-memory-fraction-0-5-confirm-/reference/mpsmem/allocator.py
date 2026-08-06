import torch

def enforce_memory_fraction(fraction: float) -> None:
    if hasattr(torch, "mps") and hasattr(torch.mps, "set_per_process_memory_fraction"):
        torch.mps.set_per_process_memory_fraction(fraction)

def check_allocation_safety(requested_bytes: int, total_device_bytes: int, fraction: float) -> dict:
    allowed = int(total_device_bytes * fraction)
    will_oom = requested_bytes > allowed
    return {
        "allowed_bytes": allowed,
        "requested_bytes": requested_bytes,
        "will_oom": will_oom
    }
