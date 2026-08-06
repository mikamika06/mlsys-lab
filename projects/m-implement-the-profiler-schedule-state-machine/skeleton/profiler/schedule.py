def get_step_action(step: int, skip_first: int, wait: int, warmup: int, active: int, repeat: int = 0) -> str:
    """Determine profiler action string for step."""
    raise NotImplementedError


def schedule_summary(total_steps: int, skip_first: int, wait: int, warmup: int, active: int, repeat: int = 0) -> dict:
    """Compute schedule step breakdown and active ranges."""
    raise NotImplementedError
