import numpy as np


def generate_test_cases():
    np.random.seed(42)
    cases = []
    for i in range(5):
        fp32_out = np.random.randn(10, 64).astype(np.float32)
        int8_out = fp32_out + 0.01 * np.random.randn(10, 64).astype(np.float32)
        fp32_size = 1024 * 1024 * (i + 4)
        int8_size = fp32_size // 4
        fp32_latency = 15.5 + i * 2.0
        int8_latency = fp32_latency / (1.5 + 0.1 * i)
        cases.append({
            "fp32_out": fp32_out,
            "int8_out": int8_out,
            "fp32_size": fp32_size,
            "int8_size": int8_size,
            "fp32_latency": fp32_latency,
            "int8_latency": int8_latency
        })
    return cases


CASES = generate_test_cases()


def compute_relative_error(fp32_out, int8_out):
    diff = np.linalg.norm(int8_out - fp32_out)
    base = np.linalg.norm(fp32_out)
    return float(diff / (base + 1e-8))


def compute_size_reduction(fp32_bytes, int8_bytes):
    return float(fp32_bytes / (int8_bytes + 1e-8))


def compute_latency_gain(fp32_ms, int8_ms):
    return float(fp32_ms / (int8_ms + 1e-8))
