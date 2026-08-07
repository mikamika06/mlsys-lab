def make_qx_quants(x: list[float], w: list[float], nmax: int) -> tuple[int, list[int]]:
    """Importance-weighted candidate-scale search (ggml-style make_qx_quants).

    x: (n,) float64 block. w: (n,) float64 positive importance weights.
    nmax: positive int, codes in [-nmax, nmax].

    If x is all zero, return (-1, zeros). Otherwise build 31 candidate
    scales d_k = (max(|x|)/nmax) * (1 + k/32) for k = -15..15
    (idx = k+15), compute each candidate's clipped rounded codes and its
    importance-weighted squared error sum(w * (x - d*codes)**2), and
    return (best_idx, best_codes) for the first candidate achieving the
    minimum error.
    """
    raise NotImplementedError('your code here')
