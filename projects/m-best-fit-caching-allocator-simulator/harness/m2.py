import ref


def check(workdir):
    from allocator.metrics import run_trace

    out = {
        "peak_allocated_match": 0.0,
        "peak_reserved_match": 0.0,
        "fragmentation_match": 0.0,
        "deltas_match": 0.0,
    }

    all_peak_alloc = True
    all_peak_res = True
    all_frag = True
    all_deltas = True

    for i, trace in enumerate(ref.TRACES):
        ref_res = ref.get_ref_metrics(i)
        try:
            got_res = run_trace(trace)
        except Exception as e:
            out["_note"] = f"Trace {i} raised error: {e}"
            return out

        if got_res.get("peak_allocated") != ref_res["peak_allocated"]:
            all_peak_alloc = False
        if got_res.get("peak_reserved") != ref_res["peak_reserved"]:
            all_peak_res = False
        if got_res.get("peak_fragmentation") != ref_res["peak_fragmentation"]:
            all_frag = False
        if got_res.get("reserved_deltas") != ref_res["reserved_deltas"]:
            all_deltas = False

    if all_peak_alloc:
        out["peak_allocated_match"] = 1.0
    if all_peak_res:
        out["peak_reserved_match"] = 1.0
    if all_frag:
        out["fragmentation_match"] = 1.0
    if all_deltas:
        out["deltas_match"] = 1.0

    return out
