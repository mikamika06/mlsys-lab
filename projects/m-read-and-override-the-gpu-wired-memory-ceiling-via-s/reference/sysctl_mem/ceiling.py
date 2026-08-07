def predict_wired_limit_mb(memsize_bytes: int) -> int:
    """Predict default GPU wired memory ceiling in MB from hw.memsize in bytes."""
    mem_gb = memsize_bytes / (1024 ** 3)
    if mem_gb <= 16:
        ratio = 0.65
    elif mem_gb <= 32:
        ratio = 0.70
    elif mem_gb <= 64:
        ratio = 0.75
    else:
        ratio = 0.80
    total_mb = memsize_bytes // (1024 * 1024)
    return int(total_mb * ratio)


def generate_sysctl_override(memsize_bytes: int, target_percentage: float) -> str:
    """Generate sysctl command to override iogpu.wired_mem_limit_mb."""
    if not (50.0 <= target_percentage <= 95.0):
        raise ValueError("Target percentage must be between 50.0 and 95.0")
    total_mb = memsize_bytes // (1024 * 1024)
    target_mb = int(total_mb * (target_percentage / 100.0))
    return f"sudo sysctl iogpu.wired_mem_limit_mb={target_mb}"
