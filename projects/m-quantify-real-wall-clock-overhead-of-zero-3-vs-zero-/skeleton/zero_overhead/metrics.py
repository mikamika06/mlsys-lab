def compute_overhead_breakdown(zero2_summary, zero3_summary):
    """Compute wall-clock overhead, relative slowdown, and component breakdown."""
    raise NotImplementedError


def evaluate_zero3_efficiency(zero2_summary, zero3_summary, max_allowed_slowdown_pct):
    """Determine if ZeRO-3 overhead is within an acceptable threshold."""
    raise NotImplementedError
