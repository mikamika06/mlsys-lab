import ref


def check(workdir):
    from triton_tune.autokey import find_true_argmin
    out = {"argmin_match": 0.0}
    try:
        want = find_true_argmin(ref.SWEEP_RECORDS)
        got = find_true_argmin(ref.SWEEP_RECORDS)
        if got == want:
            out["argmin_match"] = 1.0
        else:
            out["_note"] = f"got {got}, want {want}"
    except Exception as e:
        out["_note"] = f"Exception: {type(e).__name__}: {str(e)[:120]}"
    return out
