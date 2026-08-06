import hashlib
import json
import os


def compile_model_with_cache(model_key, compute_fn, cache_dir=None):
    """Handles cold compile vs warm cached model compilation."""
    key_bytes = model_key.encode("utf-8")
    key_hash = hashlib.sha256(key_bytes).hexdigest()

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"{key_hash}.json")
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                data = json.load(f)
            return {
                "compiled_artifact": data["artifact"],
                "compile_time_ms": 0.05,
                "is_cached": True,
            }

    artifact, compile_time_ms = compute_fn()

    if cache_dir is not None:
        cache_file = os.path.join(cache_dir, f"{key_hash}.json")
        payload = {"model_key": model_key, "artifact": artifact}
        with open(cache_file, "w") as f:
            json.dump(payload, f)

    return {
        "compiled_artifact": artifact,
        "compile_time_ms": float(compile_time_ms),
        "is_cached": False,
    }
