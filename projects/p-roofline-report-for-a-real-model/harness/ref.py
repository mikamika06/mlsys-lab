class KernelRun:
    def __init__(self, name: str, flops: int, bytes_accessed: int, time_ms: float):
        self.name = name
        self.flops = flops
        self.bytes_accessed = bytes_accessed
        self.time_ms = time_ms


class Hardware:
    def __init__(self, peak_gflops: float, peak_gbps: float):
        self.peak_gflops = peak_gflops
        self.peak_gbps = peak_gbps


def get_test_hw():
    return Hardware(19500.0, 1555.0)


def get_test_kernels():
    return [
        KernelRun("gemm_1", 2000000000, 10000000, 1.2),
        KernelRun("elem_1", 100000, 4000000, 0.8),
        KernelRun("gemm_2", 8000000000, 100000000, 5.0),
        KernelRun("softmax", 500000, 8000000, 1.5)
    ]


def ref_arithmetic_intensity(flops: int, bytes_accessed: int) -> float:
    if bytes_accessed == 0:
        return float('inf')
    return flops / bytes_accessed


def ref_kernel_performance(flops: int, time_ms: float) -> float:
    if time_ms == 0:
        return 0.0
    return flops / (time_ms * 1e6)


def ref_roofline_ceiling(hw, intensity: float) -> float:
    return min(hw.peak_gflops, intensity * hw.peak_gbps)


def ref_optimization_potential(hw, kernel) -> float:
    intensity = ref_arithmetic_intensity(kernel.flops, kernel.bytes_accessed)
    roof = ref_roofline_ceiling(hw, intensity)
    if roof == 0:
        return 0.0
    best_time = (kernel.flops / 1e6) / roof
    return max(0.0, kernel.time_ms - best_time)


def ref_predict_total_time(hw, kernels) -> float:
    t = 0.0
    for k in kernels:
        intensity = ref_arithmetic_intensity(k.flops, k.bytes_accessed)
        roof = ref_roofline_ceiling(hw, intensity)
        if roof > 0:
            t += (k.flops / 1e6) / roof
        else:
            t += k.time_ms
    return t
