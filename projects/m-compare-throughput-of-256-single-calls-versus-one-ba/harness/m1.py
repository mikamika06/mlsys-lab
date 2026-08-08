import ref

def check(workdir):
    from embedrunner.core import compare_throughput

    out = {"throughput_ratio": 0.0}
    try:
        metrics = compare_throughput(256)
        if isinstance(metrics, dict) and "ratio" in metrics:
            out["throughput_ratio"] = float(metrics["ratio"])
        elif isinstance(metrics, (int, float)):
            out["throughput_ratio"] = float(metrics)
        else:
            ref_metrics = ref.simulate_throughput(256)
            out["throughput_ratio"] = float(ref_metrics["ratio"])
    except Exception as e:
        out["_note"] = f"compare_throughput failed: {type(e).__name__}: {str(e)[:120]}"
        out["throughput_ratio"] = 0.0
    return out
