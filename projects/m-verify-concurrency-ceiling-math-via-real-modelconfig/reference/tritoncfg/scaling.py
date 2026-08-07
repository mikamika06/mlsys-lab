def compute_scaling_efficiency(runs: list[dict], threshold: float = 0.85) -> dict:
    base_run = min(runs, key=lambda x: x["instances"])
    base_k = base_run["instances"]
    base_tp = base_run["throughput"]

    per_inst = {}
    sat = None
    sorted_runs = sorted(runs, key=lambda x: x["instances"])

    for r in sorted_runs:
        k = r["instances"]
        tp = r["throughput"]
        ratio = k / base_k
        ideal_tp = base_tp * ratio
        eff = round(tp / ideal_tp, 4)
        speedup = round(tp / base_tp, 4)
        tp_per_inst = round(tp / k, 4)

        per_inst[k] = {
            "speedup": speedup,
            "scaling_efficiency": eff,
            "throughput_per_instance": tp_per_inst,
        }

        if k > base_k and eff < threshold and sat is None:
            sat = k

    return {
        "baseline_instances": base_k,
        "baseline_throughput": base_tp,
        "per_instance": per_inst,
        "saturation_instances": sat,
    }
