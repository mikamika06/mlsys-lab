import ref


def check(workdir):
    from threadperf.metrics import compute_performance_metrics

    out = {"metrics_match": 0.0, "throughput_valid": 0.0}
    want = ref.compute_metrics(ref.RUNS)
    try:
        got = compute_performance_metrics(ref.RUNS)
    except Exception as e:
        out["_note"] = f"metrics raised exception: {type(e).__name__}: {str(e)[:100]}"
        return out

    if not isinstance(got, dict):
        out["_note"] = f"expected dict, got {type(got)}"
        return out

    if abs(got.get("throughput_ratio", 0) - want["throughput_ratio"]) < 1e-5:
        out["metrics_match"] = 1.0
    else:
        out["_note"] = f"ratio got {got.get('throughput_ratio')}, want {want['throughput_ratio']}"

    if got.get("avg_openvino", 0) > 0 and got.get("avg_onnxruntime", 0) > 0:
        out["throughput_valid"] = 1.0

    return out
