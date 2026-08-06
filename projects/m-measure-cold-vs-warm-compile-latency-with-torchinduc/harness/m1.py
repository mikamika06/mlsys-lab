import ref
import tempfile

def check(workdir):
    from compcache.measurement import measure_compile_latencies
    out = {"latency_ratio_valid": 0.0}
    with tempfile.TemporaryDirectory() as d:
        try:
            cold, warm = measure_compile_latencies(d)
            if warm < cold:
                out["latency_ratio_valid"] = 1.0
            else:
                out["_note"] = f"warm ({warm}) not less than cold ({cold})"
        except Exception as e:
            out["_note"] = f"error: {type(e).__name__}: {str(e)[:120]}"
    return out
