def horner_eval(coeffs: list, x: float) -> float:
    """Evaluate p(x) = sum(coeffs[i] * x**i) via Horner's rule: one
    multiply and one add per coefficient, working down from the highest
    degree, never computing x**i directly.
    """
    raise NotImplementedError('your code here')
