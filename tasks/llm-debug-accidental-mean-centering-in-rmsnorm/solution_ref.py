import math

def rms_norm(x: list[list[float]], eps: float = 1e-5) -> list[list[float]]:
    """
    Correct RMSNorm implementation.

    Parameters
    ----------
    x : list[list[float]]
        Input 2-D list of shape (B, D).
    eps : float, optional
        Small constant added to the denominator for numerical stability.
        Default is 1e-5.

    Returns
    -------
    list[list[float]]
        Normalized list with the same shape.
    """
    result = []
    for row in x:
        sum_sq = 0.0
        for val in row:
            sum_sq += val * val
        mean_sq = sum_sq / len(row) if len(row) > 0 else 0.0
        rms = math.sqrt(mean_sq + eps)
        new_row = [val / rms for val in row]
        result.append(new_row)
    return result
