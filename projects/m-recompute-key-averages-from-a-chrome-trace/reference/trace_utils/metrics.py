from .parser import parse_key_averages


def find_largest_regression(baseline_trace, current_trace):
    base_avg = parse_key_averages(baseline_trace)
    curr_avg = parse_key_averages(current_trace)

    base_map = {item["name"]: item["self_us"] for item in base_avg}
    curr_map = {item["name"]: item["self_us"] for item in curr_avg}

    max_diff = -1.0
    worst_name = None

    all_names = set(base_map.keys()).union(set(curr_map.keys()))
    for name in all_names:
        b_val = base_map.get(name, 0.0)
        c_val = curr_map.get(name, 0.0)
        diff = c_val - b_val
        if diff > max_diff:
            max_diff = diff
            worst_name = name

    return {"name": worst_name, "regression_us": max_diff}
