import ref

def check(workdir):
    from pyrkv.allocation import compute_pyramidal_allocation
    out = {"allocation_match": 0.0}
    try:
        got = compute_pyramidal_allocation(ref.NUM_LAYERS, ref.TOTAL_BUDGET, ref.MIN_BUDGET)
        want = ref.get_reference_allocation()
        if got == want and sum(got) == ref.TOTAL_BUDGET:
            out["allocation_match"] = 1.0
        else:
            out["_note"] = f"got allocation {got}, want {want}"
    except Exception as e:
        out["_note"] = f"error: {type(e).__name__}: {str(e)}"
    return out
