import ref

def check(workdir):
    from memprof.oom import find_largest_live_allocation
    snapshot = ref.generate_oom_snapshot(seed=456)
    want = ref.find_largest_live_allocation(snapshot)
    try:
        got = find_largest_live_allocation(snapshot)
    except Exception as e:
        return {"largest_alloc_match": 0.0, "_note": f"raised exception: {e}"}

    match = 1.0 if (got == want or (got and want and got.get("id") == want.get("id"))) else 0.0
    return {"largest_alloc_match": float(match)}
