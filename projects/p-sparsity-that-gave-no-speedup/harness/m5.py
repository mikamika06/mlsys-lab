def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    import ref
    from sparse_eval.roofline import evaluate_workload_performance

    res = {
        "proves_no_speedup_for_small_batch": 0.0,
        "demonstrates_speedup_for_large_batch": 0.0,
        "proof_metrics_valid": 0.0,
    }

    hw = ref.hw_config()
    workloads = ref.workloads()

    for wl in workloads:
        ev = evaluate_workload_performance(wl["shape"], wl["is_24"], hw["peak_tflops"], hw["bandwidth_gbps"])
        if not wl["expected_speedup"] and not ev["has_speedup"] and ev["effective_speedup"] <= 1.05:
            res["proves_no_speedup_for_small_batch"] = 1.0
        elif wl["expected_speedup"] and ev["has_speedup"] and ev["effective_speedup"] >= 1.3:
            res["demonstrates_speedup_for_large_batch"] = 1.0

        if "reason" in ev and "effective_speedup" in ev:
            res["proof_metrics_valid"] = 1.0

    return res
