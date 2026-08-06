def detect_drift(memory_series):
    """Detect per-step upward drift."""
    if len(memory_series) < 2:
        return {"has_drift": False, "slope": 0.0}
    n = len(memory_series)
    x = list(range(n))
    mean_x = sum(x) / n
    mean_y = sum(memory_series) / n
    num = sum((x[i] - mean_x) * (memory_series[i] - mean_y) for i in range(n))
    den = sum((x[i] - mean_x) ** 2 for i in range(n))
    slope = num / den if den != 0 else 0.0
    has_drift = slope > 0.001 and (memory_series[-1] > memory_series[0])
    return {"has_drift": bool(has_drift), "slope": float(slope)}
