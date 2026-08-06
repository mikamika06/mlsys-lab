def stable_variance(x: list[float]) -> float:
    n = len(x)
    if n == 0:
        return 0.0
    mean = sum(x) / n
    dev_sq_sum = 0.0
    for val in x:
        diff = val - mean
        dev_sq_sum += diff * diff
    return float(dev_sq_sum / n)
