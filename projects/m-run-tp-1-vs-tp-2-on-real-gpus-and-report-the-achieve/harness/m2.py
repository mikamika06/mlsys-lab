import ref

def check(workdir):
    from tpscaling.analysis import compute_scaling_efficiency
    out = {"efficiency_match": 0.0, "throughput_ratio_match": 0.0}
    cfg = ref.CONFIGS[0]
    r1 = ref.run_benchmark(1, cfg)
    r2 = ref.run_benchmark(2, cfg)
    want = ref.compute_scaling_efficiency(r1, r2)
    try:
        got = compute_scaling_efficiency(r1, r2)
    except Exception as e:
        out["_note"] = f"raised {type(e).__name__}: {str(e)[:100]}"
        return out

    if isinstance(got, dict):
        if abs(got.get("throughput_ratio", 0.0) - want["throughput_ratio"]) < 1e-5:
            out["throughput_ratio_match"] = 1.0
        if abs(got.get("scaling_efficiency", 0.0) - want["scaling_efficiency"]) < 1e-5:
            out["efficiency_match"] = 1.0
    if "_note" not in out and (out["throughput_ratio_match"] == 0.0 or out["efficiency_match"] == 0.0):
        out["_note"] = f"got {got}, reference {want}"
    return out
