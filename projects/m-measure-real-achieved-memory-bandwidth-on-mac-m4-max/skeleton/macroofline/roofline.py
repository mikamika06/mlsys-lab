def arithmetic_intensity(m: int, n: int, k: int, itemsize: int = 2) -> float:
    """Calculate arithmetic intensity in FLOPs per byte."""
    raise NotImplementedError


def attainable_gflops(ai: float, peak_gflops: float, peak_gbps: float = 546.0) -> float:
    """Calculate attainable performance in GFLOP/s based on roofline bound."""
    raise NotImplementedError


def fit_empirical_roofline(profile_data: list[dict], peak_gbps: float = 546.0) -> dict:
    """Fit empirical roofline model on profiled matrix multiplication runs."""
    raise NotImplementedError
