import sys
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    from profoverhead.measure import measure_option_overheads

    out = {"overheads_matched": 0.0}
    want = ref.measure_option_overheads(ref.BENCHMARK_SAMPLES)
    got = measure_option_overheads(ref.BENCHMARK_SAMPLES)

    ok = True
    if not isinstance(got, dict):
        out["_note"] = f"expected dict, got {type(got).__name__}"
        return out

    for k in ["record_shapes", "with_stack", "combined"]:
        if k not in got:
            ok = False
            out["_note"] = f"missing key '{k}' in output"
            break
        if abs(float(got[k]) - want[k]) > 1e-4:
            ok = False
            out["_note"] = f"key '{k}': expected {want[k]}, got {got[k]}"
            break

    if ok:
        out["overheads_matched"] = 1.0
    return out
