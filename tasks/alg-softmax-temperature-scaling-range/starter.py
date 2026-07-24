import math

def compute_softmax(logits: list[float], temperatures: list[float]) -> list[list[float]]:
    # Unstable implementation
    result = []
    for T in temperatures:
        scaled = [z / T for z in logits]
        exp_vals = [math.exp(z) for z in scaled] # May overflow
        sum_exp = sum(exp_vals)
        result.append([e / sum_exp for e in exp_vals])
    return result
