def wanda_mask(W: list[list[float]], col_norms: list[float], keep_ratio: float) -> list[list[bool]]:
    """Wanda pruning mask: per output-row, keep the top `keep_ratio` fraction
    of weights ranked by |W_ij| * col_norms[j].

    W: shape (rows, cols).
    col_norms: shape (cols,), per-input-channel activation L2 norm.
    keep_ratio: fraction of each row's columns to keep (True in the mask).

    Returns a boolean array, same shape as W, True where the weight is kept.
    """
    raise NotImplementedError('your code here')
