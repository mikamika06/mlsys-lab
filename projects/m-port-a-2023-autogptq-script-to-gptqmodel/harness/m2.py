import ref


def check(workdir):
    from gptq_port.oracle import check_compatibility

    out = {"oracle_matched": 0.0, "tests": float(len(ref.ORACLE_TESTS))}
    ok = 0
    for i, (cfg, runtime, expected) in enumerate(ref.ORACLE_TESTS):
        got = check_compatibility(cfg, runtime)
        if got == expected:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"test {i} ({runtime}): got {got}, expected {expected}"
    out["oracle_matched"] = float(ok)
    return out
