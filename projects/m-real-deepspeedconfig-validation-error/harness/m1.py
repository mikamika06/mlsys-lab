import ref


def check(workdir):
    from dsdiag.config import validate_config
    out = {"configs_validated": 0.0}
    ok = 0
    for cfg in ref.CONFIGS:
        try:
            res = validate_config(cfg)
            if res == ref.validate_config(cfg):
                ok += 1
        except Exception:
            pass
    out["configs_validated"] = float(ok)
    return out
