import ref

def check(workdir):
    from servebatch.boundaries import reconstruct_boundaries
    out = {"boundaries_matched": 0.0}
    ok = True
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.reconstruct_boundaries(cfg["arrivals"], cfg["max_batch"], cfg["timeout"])
        got = reconstruct_boundaries(cfg["arrivals"], cfg["max_batch"], cfg["timeout"])
        if got != want:
            ok = False
            out["_note"] = f"config {i}: got {got}, want {want}"
            break
    if ok:
        out["boundaries_matched"] = 1.0
    return out
