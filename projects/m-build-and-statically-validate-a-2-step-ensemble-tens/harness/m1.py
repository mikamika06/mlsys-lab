import ref


def check(workdir):
    from ensemble.wiring import build_and_validate_wiring

    out = {"wiring_matched": 0.0}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.build_wiring(cfg)
        got = build_and_validate_wiring(cfg)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got {got}, reference {want}"
    out["wiring_matched"] = float(ok)
    return out
