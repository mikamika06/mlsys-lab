import ref


def check(workdir):
    from runner.config import resolve_default_context

    out = {"defaults_matched": 0.0}
    ok = 0
    for p in ref.PROFILES:
        want = ref.default_context(p)
        got = resolve_default_context(p)
        if got == want:
            ok += 1
    out["defaults_matched"] = float(ok)
    return out
