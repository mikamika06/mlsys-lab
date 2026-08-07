def calculate_arithmetic_intensity(M, N, K, is_sparse=False, dtype_bytes=2):
    raise NotImplementedError


def compute_roofline_bound(M, N, K, peak_tflops=312.0, bandwidth_gbps=2000.0, is_sparse=False, dtype_bytes=2):
    raise NotImplementedError


def find_breakeven_m(N, K, peak_tflops=312.0, bandwidth_gbps=2000.0, dtype_bytes=2):
    raise NotImplementedError


def evaluate_workload_performance(shape, is_24_sparse, peak_tflops=312.0, bandwidth_gbps=2000.0, dtype_bytes=2):
    raise NotImplementedError
