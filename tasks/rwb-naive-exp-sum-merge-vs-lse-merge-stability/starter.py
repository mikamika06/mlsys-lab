import math

def naive_merge(chunk_scores: list[list[float]], chunk_values: list[list[list[float]]]) -> list[float]:
    """
    Merge C chunks of RAW (unstabilized) attention scores/values into the
    combined softmax-weighted average, WITHOUT any max-subtraction:
        output = sum_i sum_j exp(L_i[j]) * V_i[j] / sum_i sum_j exp(L_i[j])
    Returns a (d,) vector. See task.md.
    """
    raise NotImplementedError('your code here')
