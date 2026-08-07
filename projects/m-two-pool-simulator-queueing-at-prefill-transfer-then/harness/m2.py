import ref


def check(workdir):
    from kvsim.validator import validate_kv_transfer_config

    out = {"configs_validated": 0.0}
    ok = 0
    for cfg in ref.CONFIGS:
        res = validate_kv_transfer_config(cfg)
        if isinstance(res, dict) and res.get("valid") is True:
            ok += 1
    for cfg in ref.INVALID_CONFIGS:
        res = validate_kv_transfer_config(cfg)
        if isinstance(res, dict) and res.get("valid") is False:
            ok += 1
    out["configs_validated"] = float(ok)
    return out
