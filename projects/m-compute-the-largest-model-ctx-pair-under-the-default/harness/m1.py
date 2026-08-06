import ref


def check(workdir):
    from runner_limits.sizing import compute_largest_pair

    out = {"sizing_matched": 0.0}
    try:
        got = compute_largest_pair(ref.MODELS, ref.DEFAULT_CEILING)
        want = ref.compute_largest_pair(ref.MODELS, ref.DEFAULT_CEILING)
        if got == want:
            out["sizing_matched"] = 1.0
        else:
            out["_note"] = f"expected {want}, got {got}"
    except Exception as e:
        out["_note"] = f"exception raised: {type(e).__name__}: {e}"
    return out
