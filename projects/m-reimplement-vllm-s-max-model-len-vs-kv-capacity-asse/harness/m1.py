import ref


def check(workdir):
    from vllmlimits.capacity import validate_capacity
    out = {"assertions_matched": 0.0}
    ok = True
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.validate_capacity(cfg["model_config"], cfg["gpu_config"], cfg["max_model_len"])
        got = validate_capacity(cfg["model_config"], cfg["gpu_config"], cfg["max_model_len"])
        if bool(want) != bool(got):
            ok = False
            out["_note"] = f"config {i}: got {got}, want {want}"
            break
    if ok:
        out["assertions_matched"] = 1.0
    return out
