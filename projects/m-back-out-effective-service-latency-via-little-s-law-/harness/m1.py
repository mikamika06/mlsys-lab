import sys
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    from capacity.trace import compute_effective_latency

    trace_data = ref.generate_trace_data()
    want = ref.ref_compute_effective_latency(trace_data)
    
    try:
        got = compute_effective_latency(trace_data)
    except Exception as e:  # noqa: BLE001
        return {"trace_matched": 0.0, "rel_err": 1.0, "_note": f"Exception raised: {e}"}

    if want == 0:
        rel_err = 0.0 if got == 0 else 1.0
    else:
        rel_err = abs(got - want) / abs(want)

    matched = 1.0 if rel_err <= 0.001 else 0.0
    out = {"trace_matched": matched, "rel_err": float(rel_err)}
    if not matched:
        out["_note"] = f"Expected effective latency {want}, got {got}"
    return out
