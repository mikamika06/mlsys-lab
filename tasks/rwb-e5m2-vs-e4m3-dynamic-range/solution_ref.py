def dynamic_range() -> tuple[float, float, float, float]:
    """
    Return the maximum and minimum normal values for E4M3 and E5M2 formats.
    The order is (max_e4m3, min_e4m3, max_e5m2, min_e5m2).
    """
    def compute(e: int, m: int) -> tuple[float, float]:
        bias = 2 ** (e - 1) - 1
        Emax = 2 ** e - 2
        max_norm = (2.0 - 2.0 ** (-m)) * 2 ** (Emax - bias)
        min_norm = 1.0 * 2 ** (1 - bias)
        return float(max_norm), float(min_norm)

    max_e4m3, min_e4m3 = compute(4, 3)
    max_e5m2, min_e5m2 = compute(5, 2)
    return (max_e4m3, min_e4m3, max_e5m2, min_e5m2)
