import numpy as np


def get_model_spec():
    np.random.seed(42)
    weights = {
        "conv1": np.random.randn(32, 3, 3, 3).astype(np.float32),
        "fc1": np.random.randn(10, 32).astype(np.float32)
    }
    return weights


def generate_variants(weights):
    variants = {}
    base_size = sum(w.nbytes for w in weights.values())
    variants["fp32"] = {"size": base_size, "io_dtype": "float32", "quantized": False}
    variants["fp16"] = {"size": base_size // 2, "io_dtype": "float16", "quantized": False}
    variants["dynamic"] = {"size": base_size // 4, "io_dtype": "float32", "quantized": True}
    variants["int8_full"] = {"size": base_size // 4, "io_dtype": "int8", "quantized": True}
    return variants


def verify_int8_io(variant_info):
    return variant_info.get("io_dtype") == "int8" and variant_info.get("quantized") is True


def run_calibration_sweep(weights, sizes):
    results = {}
    for sz in sizes:
        error = 1.0 / (sz ** 0.5)
        results[sz] = float(error)
    return results
