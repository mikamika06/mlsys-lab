import ref


def check(workdir):
    from graphops.branch import execute_with_cond_pattern
    out = {"cond_matched": 0.0}
    try:
        r1 = execute_with_cond_pattern(True, 5.0, 10.0)
        r2 = execute_with_cond_pattern(False, 5.0, 10.0)
        if r1 == 10.0 and r2 == 11.0:
            out["cond_matched"] = 1.0
        else:
            out["_note"] = f"unexpected evaluation results: {r1}, {r2}"
    except Exception as e:
        out["_note"] = f"failed with {type(e).__name__}: {str(e)[:100]}"
    return out
