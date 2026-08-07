def arithmetic_intensity(flops: int, bytes_accessed: int) -> float:
    raise NotImplementedError


def kernel_performance(flops: int, time_ms: float) -> float:
    raise NotImplementedError


def roofline_ceiling(hw, intensity: float) -> float:
    raise NotImplementedError


def optimization_potential(hw, kernel) -> float:
    raise NotImplementedError


def predict_total_time(hw, kernels: list) -> float:
    raise NotImplementedError


def generate_report(hw, kernels: list) -> list:
    raise NotImplementedError
