import sys
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    from trace_analyzer.diff import compare_profiles

    out = {"matches_reference": 0.0}

    pa = ref.parse_events(ref.generate_trace_o0())
    pb = ref.parse_events(ref.generate_trace_o99())

    want = ref.compare_profiles(pa, pb)
    got = compare_profiles(pa, pb)

    if got == want:
        out["matches_reference"] = 1.0
    else:
        out["_note"] = f"Diff mismatch. want {want}, got {got}"

    return out
