import numpy as np


def generate_quant_test_data(seed: int = 123):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-5.0, 5.0, size=(8, 128)).astype(np.float32)
    return x


def generate_attn_test_data(seed: int = 456):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((2, 64)).astype(np.float32)
    k = rng.standard_normal((32, 64)).astype(np.float32)
    v = rng.standard_normal((32, 64)).astype(np.float32)
    return q, k, v


def sample_candidates():
    return [
        {"num_ctx": 2048, "kv_type": "fp16", "recall": 0.99, "memory_bytes": 1024 * 1024 * 32},
        {"num_ctx": 4096, "kv_type": "fp16", "recall": 0.99, "memory_bytes": 1024 * 1024 * 64},
        {"num_ctx": 8192, "kv_type": "q8_0", "recall": 0.96, "memory_bytes": 1024 * 1024 * 68},
        {"num_ctx": 16384, "kv_type": "q8_0", "recall": 0.94, "memory_bytes": 1024 * 1024 * 136},
        {"num_ctx": 32768, "kv_type": "q4_0", "recall": 0.85, "memory_bytes": 1024 * 1024 * 140},
    ]
