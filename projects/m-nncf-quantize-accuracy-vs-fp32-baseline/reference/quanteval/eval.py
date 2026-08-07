import numpy as np


def compute_relative_error(fp32_output, quantized_output):
    diff = np.linalg.norm(quantized_output - fp32_output)
    norm = np.linalg.norm(fp32_output)
    return float(diff / (norm + 1e-8))


def compute_ir_size_reduction(fp32_size_bytes, int8_size_bytes):
    return float(fp32_size_bytes / (int8_size_bytes + 1e-8))


def compute_benchmark_latency_gain(fp32_latency_ms, int8_latency_ms):
    return float(fp32_latency_ms / (int8_latency_ms + 1e-8))
