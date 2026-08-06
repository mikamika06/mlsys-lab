import ref

def check(workdir):
    from grid_analyzer.core import derive_grid
    M, N = 128, 128
    ok = 0
    out = {"configs_matched": 0.0, "configs": float(len(ref.CONFIGS))}
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.derive_grid(M, N, cfg)
        try:
            got = derive_grid(M, N, cfg)
        except Exception:
            got = None
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got {got}, reference {want}"
    out["configs_matched"] = float(ok)
    return out
