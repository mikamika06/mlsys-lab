import ref

def check(workdir):
    from pyrkv.bakeoff import run_bakeoff
    out = {"bakeoff_match": 0.0}
    try:
        got = run_bakeoff(ref.PROMPTS, ref.TOTAL_BUDGET, ref.STRATEGIES)
        want = ref.get_reference_bakeoff()
        if got == want:
            out["bakeoff_match"] = 1.0
        else:
            out["_note"] = f"got bakeoff {got}, want {want}"
    except Exception as e:
        out["_note"] = f"error: {type(e).__name__}: {str(e)}"
    return out
