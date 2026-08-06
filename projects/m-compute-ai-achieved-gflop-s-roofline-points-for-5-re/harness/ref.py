import numpy as np


def get_test_data():
    np.random.seed(42)
    kernels = [
        {"name": "gemm", "flops": 1e10, "bytes": 4e7, "time_s": 0.005},
        {"name": "layernorm", "flops": 1e8, "bytes": 2e8, "time_s": 0.001},
        {"name": "softmax", "flops": 5e7, "bytes": 1.5e8, "time_s": 0.0008},
        {"name": "standard_attention", "flops": 2e9, "bytes": 4e9, "time_s": 0.04},
        {"name": "flash_attention", "flops": 2e9, "bytes": 5e8, "time_s": 0.008}
    ]
    peak_flops = 5000.0
    peak_bw = 200.0
    return kernels, peak_flops, peak_bw
