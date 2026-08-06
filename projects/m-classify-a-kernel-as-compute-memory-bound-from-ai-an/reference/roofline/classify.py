def classify_bound(ai: float, ridge_point: float) -> str:
    if ai < ridge_point:
        return "memory-bound"
    return "compute-bound"
