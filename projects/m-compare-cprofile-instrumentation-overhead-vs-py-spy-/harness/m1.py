import ref


def check(workdir):
    from profiler_metrics.overhead import compute_throughput_ratio

    want = ref.compute_throughput_ratio(ref.BASELINE_TRACES, ref.CPROFILE_TRACES, ref.PYSPY_TRACES)
    try:
        got = compute_throughput_ratio(ref.BASELINE_TRACES, ref.CPROFILE_TRACES, ref.PYSPY_TRACES)
    except Exception as e:
        return {"throughput_ratio": 0.0, "_note": f"raised {type(e).__name__}: {str(e)[:120]}"}

    if got is None:
        return {"throughput_ratio": 0.0, "_note": "returned None"}

    diff = abs(got - want)
    ok = 1.0 if diff < 1e-5 else 0.0
    out = {"throughput_ratio": ok}
    if ok == 0.0:
        out["_note"] = f"got {got}, reference {want}"
    return out
