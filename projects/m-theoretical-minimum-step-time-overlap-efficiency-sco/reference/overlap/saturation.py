"""Empirical bucket-size overlap saturation analysis."""


def find_saturation_point(bucket_profiles):
    best_b = None
    best_val = float("inf")
    for p in bucket_profiles:
        val = p["total_step_time"]
        if val < best_val:
            best_val = val
            best_b = p["bucket_size_mb"]
    return {"saturation_bucket_mb": best_b, "min_step_time": best_val}
