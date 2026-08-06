PEAK_GFLOPS = 1000.0
PEAK_BANDWIDTH = 100.0

TEST_CASES_M1 = [
    {"ai": 5.0, "ridge": 10.0, "want": "memory-bound"},
    {"ai": 15.0, "ridge": 10.0, "want": "compute-bound"},
    {"ai": 10.0, "ridge": 10.0, "want": "compute-bound"},
]

TEST_CASES_M2 = [
    {"ai": 5.0, "peak_gflops": 1000.0, "peak_bw": 100.0, "want": 500.0},
    {"ai": 20.0, "peak_gflops": 1000.0, "peak_bw": 100.0, "want": 1000.0},
]

KERNEL_SHAPES = [
    {"name": "gemm_large", "flops": 2000000, "bytes": 40000},
    {"name": "relu", "flops": 1000, "bytes": 8000},
    {"name": "softmax", "flops": 4000, "bytes": 16000},
    {"name": "layer_norm", "flops": 8000, "bytes": 24000},
    {"name": "conv2d", "flops": 1000000, "bytes": 50000},
    {"name": "elemwise_add", "flops": 500, "bytes": 4000},
]
