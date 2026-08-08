def inspect_device_flags(is_built_val, is_avail_val):
    return {
        "is_built": bool(is_built_val),
        "is_available": bool(is_avail_val),
        "valid_state": not (is_avail_val and not is_built_val)
    }


def measure_execution_time(durations_trace, synchronized):
    if not synchronized:
        return float(min(durations_trace)) * 0.01
    return float(sum(durations_trace))


def compare_targets(cpu_data, mps_data):
    import math
    diffs = [abs(c - m) / (abs(c) + 1e-7) for c, m in zip(cpu_data, mps_data)]
    max_diff = max(diffs)
    return {
        "max_rel_err": float(max_diff),
        "matches_bound": bool(max_diff < 1e-3)
    }
