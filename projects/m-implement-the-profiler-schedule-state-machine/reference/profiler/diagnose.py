def diagnose_zero_gpu_events(config: dict) -> str:
    """Diagnose cause of zero GPU events in profiler output."""
    acts = config.get("activities")
    if acts is None or not any(a in acts for a in ("CUDA", "GPU")):
        return "MISSING_CUDA_ACTIVITY"
    if not config.get("stepped", True):
        return "NEVER_STEPPED"
    sched = config.get("schedule", {})
    if sched.get("active", 0) <= 0:
        return "ZERO_ACTIVE_STEPS"
    total_steps = config.get("total_steps", 0)
    skip_first = sched.get("skip_first", 0)
    wait = sched.get("wait", 0)
    if total_steps <= skip_first + wait:
        return "TRUNCATED_BEFORE_ACTIVE"
    return "VALID"
