def reconcile_timing(rows: list, wall_clock_ms: float) -> dict:
    total_kernel_time = sum(r.get("time_ms", 0.0) for r in rows)
    discrepancy = wall_clock_ms - total_kernel_time
    ratio = total_kernel_time / wall_clock_ms if wall_clock_ms > 0 else 0.0
    return {
        "total_kernel_time_ms": total_kernel_time,
        "wall_clock_ms": wall_clock_ms,
        "discrepancy_ms": discrepancy,
        "ratio": ratio
    }
