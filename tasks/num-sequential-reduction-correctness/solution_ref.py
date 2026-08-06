def sequential_sum(values: list[float]) -> float:
    total = float(0.0)
    for value in values:
        total = float(total + float(value))
    return float(total)
