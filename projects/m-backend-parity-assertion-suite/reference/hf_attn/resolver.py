def get_valid_backends(config, env):
    res = []
    if env.get("has_flash", False) and config.get("is_decoder", True) and env.get("dtype") in ("float16", "bfloat16"):
        res.append("flash_attention_2")
    if env.get("torch_version", 0.0) >= 2.1:
        res.append("sdpa")
    res.append("eager")
    return res
