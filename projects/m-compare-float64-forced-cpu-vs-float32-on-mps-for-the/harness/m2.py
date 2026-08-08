import ref


def check(workdir):
    from mps_bench.core import measure_execution_time, compare_targets
    out = {"timing_verified": 0.0}
    try:
        trace = ref.TIMING_TRACES[0]
        unsync_val = measure_execution_time(trace, synchronized=False)
        sync_val = measure_execution_time(trace, synchronized=True)

        cpu_v = ref.CPU_REFERENCE_VECTORS[0]
        mps_v = ref.MPS_REFERENCE_VECTORS[0]
        comp = compare_targets(cpu_v, mps_v)

        if unsync_val < sync_val and comp.get("matches_bound") is True:
            out["timing_verified"] = 1.0
    except Exception as e:
        out["_note"] = str(e)
    return out
