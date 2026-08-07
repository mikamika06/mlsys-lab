def nf4_levels() -> list[float]:
    """Return the 16 NF4 level values as a list of floats."""
    from scipy.stats import norm
    q = [norm.ppf((i + 0.5) / 16) for i in range(16)]
    max_val = abs(q[0])
    for i in range(1, 16):
        v = abs(q[i])
        if v > max_val:
            max_val = v
    q_norm = [x / max_val for x in q]
    min_idx = 0
    min_val = abs(q_norm[0])
    for i in range(1, 16):
        v = abs(q_norm[i])
        if v < min_val:
            min_val = v
            min_idx = i
    q_norm[min_idx] = 0.0
    return q_norm
