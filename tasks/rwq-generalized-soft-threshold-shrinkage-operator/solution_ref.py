import math

def generalized_soft_threshold(x: list[float], beta: float, p: float) -> list[float]:
    """Generalized soft‑thresholding for the $L_p$ quasi‑norm ($0<p\le1$)."""
    out = []
    for val in x:
        abs_val = val if val >= 0.0 else -val
        if abs_val == 0.0:
            thresh = 0.0
        else:
            thresh = beta * math.pow(abs_val, p - 1.0)
        diff = abs_val - thresh
        shrunk_abs = diff if diff > 0.0 else 0.0
        sign_val = 1.0 if val > 0.0 else (-1.0 if val < 0.0 else 0.0)
        out.append(sign_val * shrunk_abs)
    return out
