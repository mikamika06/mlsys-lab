import ref


def check(workdir):
    from benchopt.analysis import identify_confounded_parameter

    out = {"confounder_identified": 0.0}
    try:
        res = identify_confounded_parameter(ref.RUN_A, ref.RUN_B)
        if res == "threads":
            out["confounder_identified"] = 1.0
        else:
            out["_note"] = f"Expected 'threads', got {res}"
    except Exception as e:
        out["_note"] = f"Error during execution: {type(e).__name__}: {str(e)[:120]}"
    return out
