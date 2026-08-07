from zerooverhead.logs import aggregate_stage_times, filter_warmup_steps


def calculate_zero3_overhead(z2_records, z3_records, warmup_steps=5):
    z2_clean = filter_warmup_steps(z2_records, warmup_steps=warmup_steps)
    z3_clean = filter_warmup_steps(z3_records, warmup_steps=warmup_steps)

    z2_agg = aggregate_stage_times(z2_clean)
    z3_agg = aggregate_stage_times(z3_clean)

    z2_total = z2_agg["total_step_ms"]
    z3_total = z3_agg["total_step_ms"]

    if z2_total <= 0.0:
        rel_overhead = 0.0
        slowdown_ratio = 1.0
    else:
        rel_overhead = (z3_total - z2_total) / z2_total
        slowdown_ratio = z3_total / z2_total

    param_gather_ms = z3_agg["param_gather_ms"]
    param_gather_pct = (
        (param_gather_ms / z3_total * 100.0) if z3_total > 0.0 else 0.0
    )

    return {
        "avg_step_z2_ms": z2_total,
        "avg_step_z3_ms": z3_total,
        "rel_overhead": rel_overhead,
        "slowdown_ratio": slowdown_ratio,
        "param_gather_pct": param_gather_pct,
        "net_overhead_ms": z3_total - z2_total,
    }
