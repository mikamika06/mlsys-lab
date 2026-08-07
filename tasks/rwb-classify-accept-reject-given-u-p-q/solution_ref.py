def classify_accept(u: float, p: list[float], q: list[float]) -> list[bool]:
    """Return a boolean list indicating acceptance for each token.

    Parameters
    ----------
    u : float
        Acceptance threshold.
    p, q : list[float]
        Lists of the same length.  ``q`` may contain zeros; in that case
        the corresponding ratio is treated as zero.

    Returns
    -------
    list[bool]
        Boolean list where element *i* is ``True`` iff
        ``u <= min(p[i]/q[i], 1)``.
    """
    n = len(p)
    out = []
    for i in range(n):
        qi = q[i]
        if qi == 0.0:
            ratio = 0.0
        else:
            ratio = p[i] / qi
        if ratio < 1.0:
            ratio_clamped = ratio
        else:
            ratio_clamped = 1.0
        out.append(u <= ratio_clamped)
    return out
