import math

def temperature_scale(logits: list[float], T: float) -> list[float]:
    if not logits:
        return []

    if isinstance(logits[0], list):
        result = []
        for row in logits:
            result.append(temperature_scale(row, T))
        return result

    N = len(logits)
    scaled = []
    for j in range(N):
        scaled.append(logits[j] / T)

    max_val = scaled[0]
    for j in range(1, N):
        if scaled[j] > max_val:
            max_val = scaled[j]

    exps = []
    sum_exp = 0.0
    for j in range(N):
        exp_val = math.exp(scaled[j] - max_val)
        exps.append(exp_val)
        sum_exp += exp_val

    probs = []
    for j in range(N):
        probs.append(exps[j] / sum_exp)

    return probs
