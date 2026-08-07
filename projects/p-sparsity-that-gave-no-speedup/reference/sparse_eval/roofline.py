from sparse_eval.benchmark import simulate_kernel_metrics
from sparse_eval.dispatch import select_execution_path


def calculate_arithmetic_intensity(M: int, N: int, K: int, is_sparse: bool = False, dtype_bytes: int = 2) -> float:
    path = "sparse_24_tensor_core" if is_sparse else "dense_unsupported_pattern"
    m = simulate_kernel_metrics((M, N, K), path, dtype_bytes)
    return float(m["flops"]) / float(m["total_bytes"])


def compute_roofline_bound(M: int, N: int, K: int, peak_tflops: float = 312.0, bandwidth_gbps: float = 2000.0, is_sparse: bool = False, dtype_bytes: int = 2) -> dict:
    intensity = calculate_arithmetic_intensity(M, N, K, is_sparse, dtype_bytes)
    path = "sparse_24_tensor_core" if is_sparse else "dense_unsupported_pattern"
    m = simulate_kernel_metrics((M, N, K), path, dtype_bytes)
    time_compute = float(m["flops"]) / (peak_tflops * 1e12)
    time_memory = float(m["total_bytes"]) / (bandwidth_gbps * 1e9)
    estimated_time = max(time_compute, time_memory)
    bottleneck = "compute" if time_compute >= time_memory else "memory"
    effective_tflops = (m["flops"] / estimated_time) / 1e12
    return {
        "time_compute": time_compute,
        "time_memory": time_memory,
        "estimated_time": estimated_time,
        "bottleneck": bottleneck,
        "effective_tflops": effective_tflops,
        "intensity": intensity,
    }


def find_breakeven_m(N: int, K: int, peak_tflops: float = 312.0, bandwidth_gbps: float = 2000.0, dtype_bytes: int = 2) -> int:
    for m in range(1, 1025):
        dense_bound = compute_roofline_bound(m, N, K, peak_tflops, bandwidth_gbps, is_sparse=False, dtype_bytes=dtype_bytes)
        sparse_bound = compute_roofline_bound(m, N, K, peak_tflops, bandwidth_gbps, is_sparse=True, dtype_bytes=dtype_bytes)
        if dense_bound["estimated_time"] / sparse_bound["estimated_time"] > 1.05:
            return m
    return 1024


def evaluate_workload_performance(shape: tuple, is_24_sparse: bool, peak_tflops: float = 312.0, bandwidth_gbps: float = 2000.0, dtype_bytes: int = 2) -> dict:
    M, N, K = shape
    path = select_execution_path(shape, is_24_sparse)
    if path != "sparse_24_tensor_core":
        return {
            "has_speedup": False,
            "theoretical_speedup": 1.0,
            "effective_speedup": 1.0,
            "reason": f"Fallback to {path}",
        }
    dense_bound = compute_roofline_bound(M, N, K, peak_tflops, bandwidth_gbps, is_sparse=False, dtype_bytes=dtype_bytes)
    sparse_bound = compute_roofline_bound(M, N, K, peak_tflops, bandwidth_gbps, is_sparse=True, dtype_bytes=dtype_bytes)
    speedup = dense_bound["estimated_time"] / sparse_bound["estimated_time"]
    has_speedup = speedup > 1.1
    reason = "Compute bound speedup achieved" if has_speedup else "Memory bandwidth bound or metadata overhead limits speedup"
    return {
        "has_speedup": has_speedup,
        "theoretical_speedup": 2.0,
        "effective_speedup": speedup,
        "reason": reason,
    }
