def compute_delta(base_config, target_mode):
    out = dict(base_config)
    if target_mode == "default":
        pass
    elif target_mode == "reduce-overhead":
        out["triton.cudagraphs"] = True
    elif target_mode == "max-autotune":
        out["max_autotune"] = True
        out["triton.cudagraphs"] = True
    return {k: out[k] for k in sorted(out.keys()) if out[k] != base_config.get(k)}
