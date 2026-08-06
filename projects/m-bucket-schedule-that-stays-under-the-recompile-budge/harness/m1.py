import ref

def check(workdir):
    from recomp.buckets import compute_buckets
    shapes, budget = ref.get_fixtures()
    want = ref.compute_buckets(shapes, budget)
    try:
        got = compute_buckets(shapes, budget)
    except Exception as e:
        return {"buckets_matched": 0.0, "_note": f"raised {type(e).__name__}"}
    
    match = (got == want) and (len(got) <= budget)
    return {"buckets_matched": 1.0 if match else 0.0, "_note": f"got {got}, want {want}"}
