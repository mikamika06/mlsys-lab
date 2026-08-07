def repair_config_for_flash(config):
    res = config.copy()
    if res["dtype"] not in ("float16", "bfloat16"):
        res["dtype"] = "float16"

    if res["has_custom_mask"]:
        res["has_custom_mask"] = False
        res["is_causal"] = True

    valid_dims = [16, 32, 64, 128, 256]
    hd = res["head_dim"]
    if hd not in valid_dims:
        if hd > 256:
            res["head_dim"] = 256
        else:
            for v in valid_dims:
                if v >= hd:
                    res["head_dim"] = v
                    break
    return res
