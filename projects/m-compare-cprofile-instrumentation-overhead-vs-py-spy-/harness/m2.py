import ref

def check(workdir):
    from profiler_bench.ranking import rank_profiler_options
    want = ref.compute_reference_ranking()
    try:
        got = rank_profiler_options()
    except Exception as e:
        return {"ranking_match": 0.0, "_note": f"raised {type(e).__name__}"}

    match = 1.0 if got == want else 0.0
    return {"ranking_match": match, "got": str(got), "want": str(want)}
