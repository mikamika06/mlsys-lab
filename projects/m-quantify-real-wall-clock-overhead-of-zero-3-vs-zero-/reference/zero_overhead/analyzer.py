def parse_step_logs(logs):
    """Parse step timing dictionaries for ZeRO runs."""
    valid_steps = []
    for step in logs:
        if "step" in step and "total_ms" in step and "stage" in step:
            valid_steps.append({
                "step": int(step["step"]),
                "stage": int(step["stage"]),
                "forward_ms": float(step.get("forward_ms", 0.0)),
                "backward_ms": float(step.get("backward_ms", 0.0)),
                "param_allgather_ms": float(step.get("param_allgather_ms", 0.0)),
                "grad_reduce_ms": float(step.get("grad_reduce_ms", 0.0)),
                "optimizer_ms": float(step.get("optimizer_ms", 0.0)),
                "total_ms": float(step["total_ms"]),
            })
    return valid_steps


def extract_stage_summary(parsed_logs, stage):
    """Extract mean timing components for a given ZeRO stage."""
    stage_logs = [s for s in parsed_logs if s["stage"] == stage]
    if not stage_logs:
        return {}
    keys = ["forward_ms", "backward_ms", "param_allgather_ms", "grad_reduce_ms", "optimizer_ms", "total_ms"]
    n = len(stage_logs)
    summary = {"stage": stage, "count": n}
    for k in keys:
        summary[k] = sum(s[k] for s in stage_logs) / n
    return summary
