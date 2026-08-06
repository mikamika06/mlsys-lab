import ref

def check(workdir):
    from mpscompile.warmup import measure_warmup
    model = ref.SimpleModel()
    x = ref.get_test_inputs()[0]
    try:
        res = measure_warmup(model, x)
        ratio = float(res.get("ratio", 0.0))
    except Exception as e:
        return {"latency_ratio": 0.0, "_note": f"warmup measurement failed: {e}"}
    return {"latency_ratio": ratio if ratio > 0 else 0.0}
