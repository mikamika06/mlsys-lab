import ref

def check(workdir):
    from grid_analyzer.validate import compare_grids
    M, N = 256, 256
    want = ref.compare_grids(M, N, ref.CONFIGS)
    out = {"comparison_matched": 0.0}
    try:
        got = compare_grids(M, N, ref.CONFIGS)
        if got == want:
            out["comparison_matched"] = 1.0
        else:
            out["_note"] = f"got {got}, want {want}"
    except Exception as e:
        out["_note"] = f"error: {e}"
    return out
