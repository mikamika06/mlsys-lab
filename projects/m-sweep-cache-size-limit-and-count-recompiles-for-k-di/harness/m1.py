import ref

def check(workdir):
    from compiler_diag.sweep import sweep_cache_limit

    out = {"sweeps_matched": 0.0}
    total = len(ref.SHAPE_DATASETS)
    passed = 0

    for shapes, limits in zip(ref.SHAPE_DATASETS, ref.CACHE_LIMIT_SWEEPS):
        want = ref.sweep_cache_limit(shapes, limits)
        got = sweep_cache_limit(shapes, limits)
        if got == want:
            passed += 1
        elif "_note" not in out:
            out["_note"] = f"mismatch on input: got {got}, want {want}"

    if passed == total:
        out["sweeps_matched"] = 1.0

    return out
