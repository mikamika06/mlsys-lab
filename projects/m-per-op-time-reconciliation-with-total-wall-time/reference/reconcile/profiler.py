def calculate_overhead_ratio(per_op_total, total_wall_time):
    if total_wall_time <= 0:
        return 0.0
    return float((total_wall_time - per_op_total) / total_wall_time)


def reconcile_profile_times(report_data):
    ops = report_data.get("ops", [])
    per_op_total = float(sum(op.get("real_time_us", 0.0) for op in ops))
    total_wall_time = float(report_data.get("total_wall_time_us", 0.0))
    rel_err = calculate_overhead_ratio(per_op_total, total_wall_time)
    return {
        "per_op_total_us": per_op_total,
        "total_wall_time_us": total_wall_time,
        "overhead_ratio": rel_err,
        "reconciled": bool(abs(rel_err) <= 0.05)
    }
