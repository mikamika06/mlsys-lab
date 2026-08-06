import ref


def check(workdir):
    from flashdiag import validator

    out = {"validation_accuracy": 0.0}
    configs = ref.VALIDATOR_CONFIGS
    ok = 0
    for i, item in enumerate(configs):
        cfg = item["config"]
        want = item["valid"]
        try:
            got = validator.validate_config(cfg)
            is_valid = bool(got)
        except Exception:
            is_valid = False
        if is_valid == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got {is_valid}, want {want}"
    if ok == len(configs):
        out["validation_accuracy"] = 1.0
    return out
