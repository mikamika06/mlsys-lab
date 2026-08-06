import ref


def check(workdir):
    from mesh.policy import diagnose_wrap_policy

    out = {"policies_matched": 0.0}
    ok = 0
    for i, (cfg, mods) in enumerate(zip(ref.CONFIGS, ref.MODULE_SETS)):
        want = ref.diagnose_wrap_policy(mods, cfg)
        try:
            got = diagnose_wrap_policy(mods, cfg)
        except Exception:
            got = {}
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got {got}, reference {want}"
    out["policies_matched"] = float(ok)
    return out
