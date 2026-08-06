import ref


def check(workdir):
    out = {"latency_ratio": 0.0, "metrics_valid": 0.0}
    try:
        from disagg.metrics import compute_latency_ratios
    except ImportError as e:
        out["_note"] = f"Import error: {e}"
        return out

    ref_agg, ref_disagg, ref_metrics = ref.get_reference_results()
    got_metrics = compute_latency_ratios(ref_agg, ref_disagg)

    required_keys = ["agg_mean_ttft", "disagg_mean_ttft", "agg_mean_itl", "disagg_mean_itl", "latency_ratio"]
    if not all(k in got_metrics for k in required_keys):
        out["_note"] = f"Missing keys in computed metrics. Got: {list(got_metrics.keys())}"
        return out

    out["metrics_valid"] = 1.0
    out["latency_ratio"] = float(got_metrics["latency_ratio"])
    return out
