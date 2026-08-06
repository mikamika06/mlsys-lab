import ref


def check(workdir):
    from symfix.core import apply_check
    out = {"checks_matched": 0.0}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.fix_code(cfg)
        got = apply_check(cfg)
        if got == want:
            ok += 1
    out["checks_matched"] = float(ok)
    return out
