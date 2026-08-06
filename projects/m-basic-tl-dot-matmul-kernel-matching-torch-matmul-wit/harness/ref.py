import random
import torch


def get_test_matrices():
    torch.manual_seed(42)
    a = torch.randn(64, 64)
    b = torch.randn(64, 64)
    return a, b


def get_benchmark_fixtures():
    return [
        {"m": 64, "n": 64, "k": 64, "time_ms": 0.1, "cublas_time_ms": 0.05},
        {"m": 128, "n": 128, "k": 128, "time_ms": 0.4, "cublas_time_ms": 0.2}
    ]
