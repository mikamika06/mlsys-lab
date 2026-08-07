def make_recipe(model_config, scheme="fp8", block_size=16):
    return {
        "model": model_config["model_name"],
        "kv_cache_dtype": "fp8" if "fp8" in scheme else "fp16",
        "kv_cache_scheme": scheme,
        "block_size": block_size,
        "enabled": True
    }
