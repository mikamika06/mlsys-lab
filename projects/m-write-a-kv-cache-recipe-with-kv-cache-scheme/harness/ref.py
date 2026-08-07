import numpy as np

CONFIGS = [
    {"model_name": "llama-7b", "num_layers": 32, "hidden_size": 4096},
    {"model_name": "llama-13b", "num_layers": 40, "hidden_size": 5120},
    {"model_name": "llama-70b", "num_layers": 80, "hidden_size": 8192},
]


def make_recipe(model_config, scheme="fp8", block_size=16):
    return {
        "model": model_config["model_name"],
        "kv_cache_dtype": "fp8" if "fp8" in scheme else "fp16",
        "kv_cache_scheme": scheme,
        "block_size": block_size,
        "enabled": True,
    }


def build_serving_args(recipe, model_path="/models/llama"):
    args = ["--model", model_path, "--kv-cache-dtype", recipe["kv_cache_dtype"]]
    if recipe.get("kv_cache_scheme"):
        args.extend(["--kv-cache-scheme", recipe["kv_cache_scheme"]])
    args.extend(["--kv-cache-block-size", str(recipe["block_size"])])
    return args


def compare_quality(baseline_outputs, quantized_outputs):
    base = np.array(baseline_outputs, dtype=np.float32)
    quant = np.array(quantized_outputs, dtype=np.float32)
    mse = float(np.mean((base - quant) ** 2))
    max_diff = float(np.max(np.abs(base - quant)))
    return {"mse": mse, "max_diff": max_diff, "valid": mse < 0.05}
