def flag_outliers(X: list[list[float]], factor: float = 3.0) -> list[bool]:
    """
    Return a boolean mask indicating channels whose maximum absolute activation
    exceeds `factor` times the median of all channel maxima.
    """
    rows = len(X)
    cols = len(X[0]) if rows > 0 else 0

    m = []
    for j in range(cols):
        max_val = 0.0
        for i in range(rows):
            val = abs(X[i][j])
            if val > max_val:
                max_val = val
        m.append(max_val)

    sorted_m = sorted(m)
    if cols == 0:
        med = 0.0
    elif cols % 2 == 1:
        med = sorted_m[cols // 2]
    else:
        med = (sorted_m[cols // 2 - 1] + sorted_m[cols // 2]) / 2.0

    threshold = factor * med
    return [val > threshold for val in m]
