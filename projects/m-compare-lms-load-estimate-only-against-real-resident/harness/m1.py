import ref


def check(workdir):
    from lmsmem import estimate_bytes

    out = {"estimates_matched": 0.0, "cases": float(len(ref.CASES))}
    ok = 0
    for i, (cfg, ctx, bpp) in enumerate(ref.CASES):
        want = ref.estimate_bytes(cfg, ctx, bpp)
        got = estimate_bytes(cfg, ctx, bpp)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"case {i}: got {got}, reference {want}"
    out["estimates_matched"] = float(ok)
    return out
