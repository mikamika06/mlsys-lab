import ref


def check(workdir):
    from gptq_port.translate import translate_config

    out = {"configs_matched": 0.0, "configs": float(len(ref.CONFIGS))}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = {
            "quant_method": "gptq",
            "bits": cfg.get("bits", 4),
            "group_size": cfg.get("group_size", 128),
            "damp_percent": cfg.get("damp_percent", 0.1),
            "desc_act": cfg.get("desc_act", False),
            "static_groups": cfg.get("static_groups", False),
            "sym": cfg.get("sym", True),
            "true_sequential": cfg.get("true_sequential", True),
            "model_name_or_path": cfg.get("model_name_or_path", None),
            "model_file_base_name": cfg.get("model_file_base_name", "model")
        }
        got = translate_config(cfg)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got {got}, reference {want}"
    out["configs_matched"] = float(ok)
    return out
