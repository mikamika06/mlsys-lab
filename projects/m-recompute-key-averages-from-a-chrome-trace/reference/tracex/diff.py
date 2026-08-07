from tracex.parse import compute_key_averages

def largest_self_time_regression(trace_a, trace_b):
    a_stats = compute_key_averages(trace_a)
    b_stats = compute_key_averages(trace_b)
    max_reg = 0.0
    worst_name = None
    all_names = set(a_stats.keys()).union(set(b_stats.keys()))
    for name in all_names:
        a_self = a_stats.get(name, {}).get("total_self_time", 0.0)
        b_self = b_stats.get(name, {}).get("total_self_time", 0.0)
        diff = b_self - a_self
        if diff > max_reg:
            max_reg = diff
            worst_name = name
    return {"name": worst_name, "regression": float(max_reg)}
