def compare_latencies(compute_plan, measured):
    estimates = {item["op_name"]: item["estimated_cost"] for item in compute_plan.get("ops", [])}
    results = {}
    for name, meas in measured.items():
        est = estimates.get(name, 0.0)
        diff = abs(est - meas) / (meas + 1e-9)
        results[name] = {"estimated": est, "measured": meas, "rel_diff": diff}
    return results
