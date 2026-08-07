def predict_wired_limit_mb(memsize_bytes: int) -> int:
    """Predict default GPU wired memory ceiling in MB from hw.memsize in bytes."""
    raise NotImplementedError


def generate_sysctl_override(memsize_bytes: int, target_percentage: float) -> str:
    """Generate sysctl command to override iogpu.wired_mem_limit_mb."""
    raise NotImplementedError
