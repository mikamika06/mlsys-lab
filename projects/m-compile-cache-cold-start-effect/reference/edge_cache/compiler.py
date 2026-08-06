import os
import hashlib


def compute_sha256(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def compile_and_run(model_path: str, cache_dir: str, input_data: list) -> dict:
    os.makedirs(cache_dir, exist_ok=True)
    model_hash = compute_sha256(model_path)
    cache_path = os.path.join(cache_dir, f"{model_hash}.compiled")

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            compiled_weights = f.read()
        is_cold = False
        compilation_latency_ms = 2.0
    else:
        with open(model_path, "rb") as f:
            raw_weights = f.read()
        compiled_weights = bytes(b ^ 0xAA for b in raw_weights)
        with open(cache_path, "wb") as f:
            f.write(compiled_weights)
        is_cold = True
        compilation_latency_ms = 120.0

    sum_val = sum(input_data) + sum(compiled_weights[:16])
    execution_latency_ms = 5.0 + (sum_val % 3)
    total_latency_ms = compilation_latency_ms + execution_latency_ms

    return {
        "output": sum_val,
        "is_cold": is_cold,
        "latency_ms": total_latency_ms,
    }
