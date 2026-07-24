import math

def compute_softmax(logits: list[float], temperatures: list[float]) -> list[list[float]]:
    result = []
    for T in temperatures:
        scaled = [z / T for z in logits]
        max_val = max(scaled)
        exp_vals = [math.exp(z - max_val) for z in scaled]
        sum_exp = sum(exp_vals)
        result.append([e / sum_exp for e in exp_vals])
    return result
