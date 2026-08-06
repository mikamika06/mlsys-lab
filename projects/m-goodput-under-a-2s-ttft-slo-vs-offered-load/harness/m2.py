import ref

def check(workdir):
    from goodput.little import check_littles_law
    out = {"little_match": 0.0}
    try:
        want = ref.check_littles_law(ref.ARRIVALS, ref.QUEUES, ref.LATENCIES)
        got = check_littles_law(ref.ARRIVALS, ref.QUEUES, ref.LATENCIES)
        if abs(want - got) < 1e-5:
            out["little_match"] = 1.0
        else:
            out["_note"] = f"got error {got}, want {want}"
    except Exception as e:
        out["_note"] = f"exception raised: {type(e).__name__}: {str(e)[:100]}"
    return out
