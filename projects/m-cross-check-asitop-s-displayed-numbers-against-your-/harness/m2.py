import ref

def check(workdir):
    out = {"gaps_matched": 0.0}
    _, _, events, _ = ref.generate_fixtures()
    want = ref.count_gaps(events, 2000.0)

    from edgemetrics.trace import count_gaps

    try:
        got = count_gaps(events, 2000.0)
        if got == want:
            out["gaps_matched"] = 1.0
        else:
            out["_note"] = f"count_gaps got {got}, want {want}"
    except Exception as e:
        out["_note"] = f"Exception during execution: {type(e).__name__}: {str(e)}"
    return out
