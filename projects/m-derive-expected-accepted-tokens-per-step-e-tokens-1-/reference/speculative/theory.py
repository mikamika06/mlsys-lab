def expected_accepted_tokens(a, g):
    if abs(a - 1.0) < 1e-9:
        return float(g + 1)
    return float((1.0 - a**(g + 1)) / (1.0 - a))
