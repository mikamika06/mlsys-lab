import ref

def check(workdir):
    from profiler_bench.overhead import measure_overhead_ratio
    want = ref.compute_reference_overhead()
    try:
        got = measure_overhead_ratio()
    except Exception as e:
        return {"overhead_ratio_match": 0.0, "_note": f"raised {type(e).__name__}"}

    diff = abs(got - want) / max(abs(want), 1e-6)
    match = 1.0 if diff < 0.5 else 0.0
    return {"overhead_ratio_match": match, "got": float(got), "want": float(want)}
