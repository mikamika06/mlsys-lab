import ref


def check(workdir):
    from graphops.branch import analyze_compilation_behavior
    out = {"metrics_matched": 0.0}
    try:
        got = analyze_compilation_behavior(ref.RECORDS)
        want = ref.get_expected_analysis()
        if got == want:
            out["metrics_matched"] = 1.0
        else:
            out["_note"] = f"got {got}, want {want}"
    except Exception as e:
        out["_note"] = f"failed with {type(e).__name__}: {str(e)[:100]}"
    return out
