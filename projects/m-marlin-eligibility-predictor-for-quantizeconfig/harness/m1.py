import ref


def check(workdir):
    from marlin.eligibility import is_marlin_eligible

    out = {"predictions_matched": 0.0, "total": float(len(ref.CONFIGS))}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = cfg["expected_eligible"]
        got = is_marlin_eligible(cfg)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got {got}, reference {want}"
    out["predictions_matched"] = float(ok)
    return out
