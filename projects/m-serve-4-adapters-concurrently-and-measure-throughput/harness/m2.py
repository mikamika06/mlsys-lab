import ref

def check(workdir):
    from loraserve.engine import run_concurrent_batch
    from loraserve.metrics import compute_throughput_ratio
    reqs = ref.get_sample_requests() * 10
    adapters = ref.ADAPTERS
    out = {"throughput_ratio": 0.0}
    try:
        base_metrics = ref.run_base_baseline(reqs)
        multi_metrics = run_concurrent_batch(reqs, adapters)
        ratio = compute_throughput_ratio(multi_metrics, base_metrics)
        out["throughput_ratio"] = float(ratio)
    except Exception as e:
        out["_note"] = f"throughput check failed: {type(e).__name__}: {str(e)[:120]}"
    return out
