import os

SAMPLE_EXPORT = {
    "operations": [
        {"op_name": "op1", "compute_unit": "ANE"},
        {"op_name": "op2", "compute_unit": "ANE"},
        {"op_name": "op3", "compute_unit": "CPU"},
        {"op_name": "op4", "compute_unit": "GPU"},
    ]
}


def recompute_routing(profiler_export):
    ops = profiler_export.get("operations", [])
    total_ops = len(ops)
    if total_ops == 0:
        return {"ane_percentage": 0.0, "cpu_percentage": 0.0, "gpu_percentage": 0.0}
    counts = {"ANE": 0, "CPU": 0, "GPU": 0}
    for op in ops:
        target = op.get("compute_unit", "CPU")
        counts[target] = counts.get(target, 0) + 1
    return {
        "ane_percentage": round(counts["ANE"] * 100.0 / total_ops, 2),
        "cpu_percentage": round(counts["CPU"] * 100.0 / total_ops, 2),
        "gpu_percentage": round(counts["GPU"] * 100.0 / total_ops, 2),
    }


def setup_bundle(bundle_path):
    os.makedirs(bundle_path, exist_ok=True)
    with open(os.path.join(bundle_path, "model.espresso.net"), "w") as f:
        f.write("net")
    with open(os.path.join(bundle_path, "model.espresso.weights"), "w") as f:
        f.write("weights")


def verify_mlmodelc(bundle_path):
    if not os.path.isdir(bundle_path):
        return False
    metadata_path = os.path.join(bundle_path, "model.espresso.net")
    weights_path = os.path.join(bundle_path, "model.espresso.weights")
    return os.path.isfile(metadata_path) and os.path.isfile(weights_path)


def compare_latencies(compute_plan, measured):
    estimates = {item["op_name"]: item["estimated_cost"] for item in compute_plan.get("ops", [])}
    results = {}
    for name, meas in measured.items():
        est = estimates.get(name, 0.0)
        diff = abs(est - meas) / (meas + 1e-9)
        results[name] = {"estimated": est, "measured": meas, "rel_diff": diff}
    return results
