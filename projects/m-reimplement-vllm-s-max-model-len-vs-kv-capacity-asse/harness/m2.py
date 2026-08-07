import ref


def check(workdir):
    from vllmlimits.capacity import max_safe_model_len
    out = {"max_lens_matched": 0.0}
    ok = True
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.max_safe_model_len(cfg["model_config"], cfg["gpu_config"])
        got = max_safe_model_len(cfg["model_config"], cfg["gpu_config"])
        if want != got:
            ok = False
            out["_note"] = f"config {i}: got {got}, want {want}"
            break
    if ok:
        out["max_lens_matched"] = 1.0
    return out
