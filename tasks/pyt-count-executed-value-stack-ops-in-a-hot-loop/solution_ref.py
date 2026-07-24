def horner_eval(coeffs: list, x: float) -> float:
    """Evaluate a polynomial via Horner's rule: one multiply + one add per
    coefficient, working down from the highest degree. No `**`.
    """
    it = iter(reversed(coeffs))
    acc = next(it)
    for c in it:
        acc = acc * x + c
    return acc
