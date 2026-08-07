def find_argmin_config(candidate_logs: list) -> dict:
    """Find the argmin configuration from max-autotune candidate timing logs."""
    if not candidate_logs:
        return {}

    valid_candidates = [c for c in candidate_logs if c.get("status") == "OK" and "time_ms" in c]
    if not valid_candidates:
        return {}

    best = min(valid_candidates, key=lambda x: x["time_ms"])
    return {
        "config": best["config"],
        "time_ms": best["time_ms"],
        "num_candidates_evaluated": len(candidate_logs),
    }
