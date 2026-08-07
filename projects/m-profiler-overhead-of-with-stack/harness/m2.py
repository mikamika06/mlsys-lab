import ref

def check(workdir):
    from proftune.overhead import compute_overhead_ratio
    out = {"overhead_matched": 0.0}
    base = 100.0
    prof = 180.0
    want = ref.compute_overhead_ratio(base, prof, with_stack=True)
    got = compute_overhead_ratio(base, prof, with_stack=True)
    if abs(got - want) < 1e-5:
        out["overhead_matched"] = 1.0
    else:
        out["_note"] = f"got overhead ratio {got}, want {want}"
    return out
