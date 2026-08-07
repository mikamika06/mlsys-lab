def roofline_ceiling(intensity: float, hw_spec: dict) -> float:
    raise NotImplementedError


def classify_kernel(intensity: float, hw_spec: dict) -> str:
    raise NotImplementedError


def kernel_performance_bound(kernel_stats: dict, hw_spec: dict) -> dict:
    raise NotImplementedError
