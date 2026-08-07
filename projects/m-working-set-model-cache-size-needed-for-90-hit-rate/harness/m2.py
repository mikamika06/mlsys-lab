import ref


def check(workdir):
    from workset.warmup import warmup_curve

    out = {"curves_matched": 0.0, "configs": float(len(ref.CURVE_CASES))}
    ok = 0
    for i, case in enumerate(ref.CURVE_CASES):
        want = ref.warmup_curve(case["trace"], case["cache_size"])
        got = warmup_curve(case["trace"], case["cache_size"])
        if list(got) == list(want):
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"case {i}: length or values mismatch"
    out["curves_matched"] = float(ok)
    return out
