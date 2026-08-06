import math


def hillis_steele_scan(x: list[int]) -> list[int]:
    """Inclusive prefix-sum scan via the Hillis-Steele distance-doubling recurrence.

    Parameters
    ----------
    x : list of int
        Input sequence.

    Returns
    -------
    list of int
        y[i] = sum(x[0..i]), computed in ceil(log2(N)) rounds where round k
        combines each element with the one 2**k positions to its left.
    """
    y = list(x)
    n = len(y)
    if n <= 1:
        return y

    n_rounds = math.ceil(math.log2(n))
    shift = 1
    for _ in range(n_rounds):
        prev = list(y)  # read the previous round's values only
        if shift < n:
            for i in range(shift, n):
                y[i] = prev[i] + prev[i - shift]
        shift *= 2
    return y
