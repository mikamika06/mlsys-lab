import torch

def configure_memory_fraction(fraction: float) -> None:
    if hasattr(torch, "mps") and hasattr(torch.mps, "set_per_process_memory_fraction"):
        torch.mps.set_per_process_memory_fraction(fraction)

def check_oom_threshold(limit_fraction: float, total_budget_bytes: int) -> dict:
    allowed = int(total_budget_bytes * limit_fraction)
    alloc_size = int(allowed * 1.2)
    oom_triggered = False
    try:
        if alloc_size > allowed:
            raise RuntimeError("MPS out of memory")
    except RuntimeError:
        oom_triggered = True
    return {
        "allowed_bytes": allowed,
        "attempted_bytes": alloc_size,
        "oom_triggered": oom_triggered
    }
