import ref

def check(workdir):
    from loraserve.engine import run_concurrent_batch
    reqs = ref.get_sample_requests()
    adapters = ref.ADAPTERS
    out = {"adapters_active": 0.0}
    try:
        res = run_concurrent_batch(reqs, adapters)
        out["adapters_active"] = float(res.get("active_adapters", 0))
    except Exception as e:
        out["_note"] = f"run_concurrent_batch failed: {type(e).__name__}: {str(e)[:120]}"
    return out
