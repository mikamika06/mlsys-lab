import ref


def check(workdir):
    from profiler_utils.metrics import measure_trace_overhead
    traces_no, traces_with = ref.generate_traces(100)
    want = ref.measure_trace_overhead(traces_no, traces_with)
    got = measure_trace_overhead(traces_no, traces_with)
    out = {"size_ratio_match": 0.0}
    if abs(got - want) < 1e-5:
        out["size_ratio_match"] = 1.0
    else:
        out["_note"] = f"got overhead {got}, want {want}"
    return out
