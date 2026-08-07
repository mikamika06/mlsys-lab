"""Per-op vs wall clock gap module."""


def analyze_wall_clock_gap(per_op_times, wall_clock_time):
    sum_op_time = float(sum(per_op_times))
    gap = float(wall_clock_time) - sum_op_time
    overhead_ratio = gap / float(wall_clock_time) if wall_clock_time > 0 else 0.0
    return {
        "sum_op_time": sum_op_time,
        "wall_clock_time": float(wall_clock_time),
        "gap": gap,
        "overhead_ratio": overhead_ratio,
    }
