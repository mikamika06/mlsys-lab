import hashlib


def check_engine_compatibility(engine_meta, target_env):
    required_keys = ["compute_capability", "cuda_version", "trt_version"]
    for k in required_keys:
        if k not in engine_meta or k not in target_env:
            return False, f"Missing key: {k}"

    if engine_meta["compute_capability"] != target_env["compute_capability"]:
        return False, f"Compute capability mismatch: {engine_meta['compute_capability']} vs {target_env['compute_capability']}"

    if engine_meta["cuda_version"] != target_env["cuda_version"]:
        return False, f"CUDA version mismatch: {engine_meta['cuda_version']} vs {target_env['cuda_version']}"

    if engine_meta["trt_version"] != target_env["trt_version"]:
        return False, f"TensorRT version mismatch: {engine_meta['trt_version']} vs {target_env['trt_version']}"

    return True, "Compatible"


def compute_compatibility_hash(env):
    keys = ["compute_capability", "cuda_version", "trt_version"]
    s = "|".join(str(env.get(k, "")) for k in keys)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()
