import sys


def check(workdir):
    sys.path.insert(0, workdir)
    import ref
    from kvquant import measure_fused_path_penalty

    out = {"penalties_matched": 0.0}

    penalties_ok = True
    for tk, tv in ref.PENALTY_CASES:
        want = ref.measure_fused_path_penalty(tk, tv)
        got = measure_fused_path_penalty(tk, tv)
        if abs(got - want) > 1e-4:
            penalties_ok = False
            out["_note"] = f"penalty mismatch for {tk}/{tv}: want {want}, got {got}"
            break

    if penalties_ok:
        out["penalties_matched"] = 1.0

    return out
