def compute_overhead_breakdown(zero2_summary, zero3_summary):
    """Compute wall-clock overhead, relative slowdown, and component breakdown."""
    total_z2 = zero2_summary.get("total_ms", 0.0)
    total_z3 = zero3_summary.get("total_ms", 0.0)
    abs_overhead_ms = total_z3 - total_z2
    rel_slowdown = (total_z3 / total_z2) if total_z2 > 0 else 0.0
    slowdown_pct = (rel_slowdown - 1.0) * 100.0 if total_z2 > 0 else 0.0

    comm_z2 = zero2_summary.get("grad_reduce_ms", 0.0)
    comm_z3 = zero3_summary.get("param_allgather_ms", 0.0) + zero3_summary.get("grad_reduce_ms", 0.0)
    comm_overhead_ms = comm_z3 - comm_z2

    return {
        "zero2_total_ms": total_z2,
        "zero3_total_ms": total_z3,
        "abs_overhead_ms": abs_overhead_ms,
        "rel_slowdown": rel_slowdown,
        "slowdown_pct": slowdown_pct,
        "comm_overhead_ms": comm_overhead_ms,
        "param_gather_overhead_ms": zero3_summary.get("param_allgather_ms", 0.0),
    }


def evaluate_zero3_efficiency(zero2_summary, zero3_summary, max_allowed_slowdown_pct):
    """Determine if ZeRO-3 overhead is within an acceptable threshold."""
    metrics = compute_overhead_breakdown(zero2_summary, zero3_summary)
    is_efficient = metrics["slowdown_pct"] <= max_allowed_slowdown_pct
    return {
        "is_efficient": is_efficient,
        "slowdown_pct": metrics["slowdown_pct"],
        "max_allowed_slowdown_pct": float(max_allowed_slowdown_pct),
    }
