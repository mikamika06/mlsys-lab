def diagnose_cache_behavior(l2_capacity_bytes, working_set_bytes, measured_l2_hit_rate):
    is_large = working_set_bytes > l2_capacity_bytes
    near_zero = measured_l2_hit_rate < 0.05
    diagnosis = "working_set_larger_than_l2" if (is_large and near_zero) else "normal_cache_behavior"
    return {
        "is_working_set_large": is_large,
        "near_zero_hit_rate": near_zero,
        "diagnosis": diagnosis
    }
