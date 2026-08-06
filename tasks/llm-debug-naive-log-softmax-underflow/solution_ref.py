import math

def log_softmax(x):
    """Compute log(softmax(x)) along the last axis (numerically stable)."""
    if not x:
        return []
    if isinstance(x[0], (int, float)):
        max_val = max(x)
        sum_exp = sum(math.exp(v - max_val) for v in x)
        lse = math.log(sum_exp) + max_val
        return [v - lse for v in x]

    res = []
    for row in x:
        max_val = max(row)
        sum_exp = sum(math.exp(v - max_val) for v in row)
        lse = math.log(sum_exp) + max_val
        res.append([v - lse for v in row])
    return res
