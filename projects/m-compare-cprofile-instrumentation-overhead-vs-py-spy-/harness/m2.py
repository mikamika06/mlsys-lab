import ref


def check(workdir):
    from profiler_metrics.ranking import rank_profiler_flags

    want = ref.rank_profiler_flags(ref.FLAG_MEASUREMENTS)
    try:
        got = rank_profiler_flags(ref.FLAG_MEASUREMENTS)
    except Exception as e:
        return {"ranking_match": 0, "_note": f"raised {type(e).__name__}: {str(e)[:120]}"}

    match = 1 if got == want else 0
    out = {"ranking_match": match}
    if match == 0:
        out["_note"] = f"got {got}, reference {want}"
    return out
