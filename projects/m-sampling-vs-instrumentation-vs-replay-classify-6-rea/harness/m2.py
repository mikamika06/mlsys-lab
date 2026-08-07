import ref


def check(workdir):
    from profiler.taxonomy import calculate_miss_bound

    out = {"bounds_matched": 0.0, "total": float(len(ref.MISS_BOUND_CASES))}
    ok = 0
    for i, c in enumerate(ref.MISS_BOUND_CASES):
        try:
            got = calculate_miss_bound(c["interval"], c["duration"], c["total"])
            want = ref.calculate_miss_bound(c["interval"], c["duration"], c["total"])
            if abs(got - want) < 1e-5:
                ok += 1
            elif "_note" not in out:
                out["_note"] = f"case {i}: got {got}, want {want}"
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"case {i} raised {type(e).__name__}: {e}"
    out["bounds_matched"] = float(ok)
    return out
