import ref


def check(workdir):
    from recompile.tracker import count_recompilations

    out = {"recompile_counts_matched": 0.0}
    try:
        got = count_recompilations(ref.SAMPLE_LOGS)
        want = ref.oracle_count(ref.SAMPLE_LOGS)
        if got == want:
            out["recompile_counts_matched"] = 1.0
        else:
            out["_note"] = f"got {got}, want {want}"
    except Exception as e:
        out["_note"] = f"error: {type(e).__name__}: {str(e)[:100]}"
    return out
