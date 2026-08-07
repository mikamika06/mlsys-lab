import ref


def check(workdir):
    from vlmattn.config import parse_submodule_configs

    out = {"configs_matched": 0.0}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.parse_submodule_configs(cfg)
        got = parse_submodule_configs(cfg)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got {got}, reference {want}"
    out["configs_matched"] = float(ok)
    return out
