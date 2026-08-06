import ref


def check(workdir):
    from mps_diag.latency import measure_latency_cliff

    try:
        ratio = measure_latency_cliff(fallback_latency=500.0, native_latency=10.0)
        expected = ref.compute_latency_ratio(500.0, 10.0)
        if abs(ratio - expected) < 1e-5:
            return {"latency_ratio_correct": 1.0}
        return {"latency_ratio_correct": 0.0, "_note": f"got ratio {ratio}, expected {expected}"}
    except Exception as e:
        return {"latency_ratio_correct": 0.0, "_note": f"raised {type(e).__name__}: {str(e)[:100]}"}
