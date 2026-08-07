import ref

def check(workdir):
    from repack.bench import benchmark_repack
    out = {"speedup_valid": 0.0, "overhead_bounded": 0.0}
    try:
        res = benchmark_repack(ref.SAMPLE_WEIGHTS, {"avx2": True})
        if not isinstance(res, dict):
            out["_note"] = "benchmark_repack must return a dictionary"
            return out
        if "speedup" not in res or "variant" not in res:
            out["_note"] = "missing keys in benchmark result"
            return out
        if res["variant"] == "q4_0_avx2" and isinstance(res["speedup"], (int, float)):
            out["speedup_valid"] = 1.0
        if 0.01 <= res.get("speedup", 0.0) <= 1000.0:
            out["overhead_bounded"] = 1.0
        else:
            out["_note"] = f"speedup out of bounds: {res.get('speedup')}"
    except Exception as e:
        out["_note"] = f"benchmark raised exception: {str(e)[:100]}"
    return out
