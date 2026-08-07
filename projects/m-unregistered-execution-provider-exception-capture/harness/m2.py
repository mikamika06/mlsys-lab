import ref

def check(workdir):
    from ortprovider.inference import compare_overhead
    out = {"overhead_reduced": 0.0}
    class DummySession:
        pass
    metrics = compare_overhead(DummySession(), {})
    if isinstance(metrics, dict) and metrics.get("ratio", 1.0) < 1.0:
        out["overhead_reduced"] = 1.0
    else:
        out["_note"] = f"metrics returned: {metrics}"
    return out
