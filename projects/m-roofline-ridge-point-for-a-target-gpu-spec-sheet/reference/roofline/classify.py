def classify_decode(arithmetic_intensity: float, ridge_point: float) -> str:
    if arithmetic_intensity < ridge_point:
        return "memory-bound"
    return "compute-bound"
