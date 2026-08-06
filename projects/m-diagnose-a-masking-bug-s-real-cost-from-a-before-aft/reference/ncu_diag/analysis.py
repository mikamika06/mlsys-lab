def compute_masking_cost(before_metrics: dict, after_metrics: dict) -> dict:
    cycles_before = float(before_metrics.get("sm__cycles_elapsed.avg", 1000.0))
    cycles_after = float(after_metrics.get("sm__cycles_elapsed.avg", 1200.0))
    inst_before = float(before_metrics.get("inst_executed", 500000.0))
    inst_after = float(after_metrics.get("inst_executed", 600000.0))
    stall_mio_before = float(before_metrics.get("stall_mio_not_ready", 100.0))
    stall_mio_after = float(after_metrics.get("stall_mio_not_ready", 250.0))

    cycle_diff = cycles_after - cycles_before
    inst_diff = inst_after - inst_before
    stall_diff = stall_mio_after - stall_mio_before
    overhead_pct = (cycle_diff / cycles_before) * 100.0 if cycles_before > 0 else 0.0

    return {
        "cycle_diff": cycle_diff,
        "inst_diff": inst_diff,
        "stall_diff": stall_diff,
        "overhead_pct": overhead_pct
    }
