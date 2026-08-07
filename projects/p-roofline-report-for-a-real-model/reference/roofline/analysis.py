def arithmetic_intensity(flops: int, bytes_accessed: int) -> float:
    if bytes_accessed == 0:
        return float('inf')
    return flops / bytes_accessed


def kernel_performance(flops: int, time_ms: float) -> float:
    if time_ms == 0:
        return 0.0
    return flops / (time_ms * 1e6)


def roofline_ceiling(hw, intensity: float) -> float:
    return min(hw.peak_gflops, intensity * hw.peak_gbps)


def optimization_potential(hw, kernel) -> float:
    intensity = arithmetic_intensity(kernel.flops, kernel.bytes_accessed)
    roof = roofline_ceiling(hw, intensity)
    if roof == 0:
        return 0.0
    best_time_ms = (kernel.flops / 1e6) / roof
    return max(0.0, kernel.time_ms - best_time_ms)


def predict_total_time(hw, kernels: list) -> float:
    total_time = 0.0
    for k in kernels:
        intensity = arithmetic_intensity(k.flops, k.bytes_accessed)
        roof = roofline_ceiling(hw, intensity)
        if roof > 0:
            total_time += (k.flops / 1e6) / roof
        else:
            total_time += k.time_ms
    return total_time


def generate_report(hw, kernels: list) -> list:
    report = []
    for k in kernels:
        saved = optimization_potential(hw, k)
        report.append({"name": k.name, "saved_ms": saved})
    report.sort(key=lambda x: x["saved_ms"], reverse=True)
    return report
