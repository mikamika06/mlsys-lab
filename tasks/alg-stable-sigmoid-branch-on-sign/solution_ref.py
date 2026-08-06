import math

def stable_sigmoid(z: list[float]) -> list[float]:
    out = []
    for val in z:
        if val >= 0.0:
            out.append(1.0 / (1.0 + math.exp(-val)))
        else:
            exp_val = math.exp(val)
            out.append(exp_val / (1.0 + exp_val))
    return out
