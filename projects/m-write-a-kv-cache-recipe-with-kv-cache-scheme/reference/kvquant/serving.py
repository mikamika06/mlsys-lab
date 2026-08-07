def build_serving_args(recipe, model_path):
    args = ["--model", model_path, "--kv-cache-dtype", recipe["kv_cache_dtype"]]
    if recipe.get("kv_cache_scheme"):
        args.extend(["--kv-cache-scheme", recipe["kv_cache_scheme"]])
    args.extend(["--kv-cache-block-size", str(recipe["block_size"])])
    return args
