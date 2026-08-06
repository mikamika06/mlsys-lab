import hashlib
import os
import json
import numpy as np


def compute_sha256(file_path):
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def generate_test_environment(base_dir, num_files=4):
    os.makedirs(base_dir, exist_ok=True)
    manifest_files = {}
    rng = np.random.default_rng(42)
    for i in range(num_files):
        filename = f"artifact_{i}.bin"
        filepath = os.path.join(base_dir, filename)
        data = rng.integers(0, 256, size=1024, dtype=np.uint8).tobytes()
        with open(filepath, "wb") as f:
            f.write(data)
        manifest_files[filename] = hashlib.sha256(data).hexdigest()

    manifest_path = os.path.join(base_dir, "manifest.json")
    manifest_data = {"files": manifest_files}
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f)
    return manifest_path


def verify_manifest(manifest_path, root_dir):
    with open(manifest_path, "r") as f:
        data = json.load(f)
    files = data.get("files", {})
    for rel_path, expected_hash in files.items():
        full_path = os.path.join(root_dir, rel_path)
        if not os.path.exists(full_path):
            return False
        if compute_sha256(full_path) != expected_hash:
            return False
    return True


def simulate_compiler_execution(model_path, cache_dir, input_data):
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


def compute_population_p95(cohort_samples, cohort_weights):
    all_values = []
    all_weights = []
    for samples, weight in zip(cohort_samples, cohort_weights):
        all_values.extend(samples)
        all_weights.extend([weight / len(samples)] * len(samples))

    values = np.array(all_values)
    weights = np.array(all_weights)

    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]

    cum_weights = np.cumsum(weights)
    total_weight = cum_weights[-1]
    target = 0.95 * total_weight

    idx = np.searchsorted(cum_weights, target)
    return float(values[min(idx, len(values) - 1)])
