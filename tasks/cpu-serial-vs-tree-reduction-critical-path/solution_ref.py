def critical_path_lengths(n: int, latency: int) -> tuple[int, int]:
    """
    Compute the critical‑path lengths for a serial chain reduction and a balanced binary‑tree reduction.

    Parameters
    ----------
    n : int
        Number of operands. Must be >= 1.
    latency : int
        Latency (in cycle units) of each binary operation. Must be > 0.

    Returns
    -------
    tuple[int, int]
        A pair ``(serial, tree)`` where:
            * ``serial`` is the critical‑path length for a linear chain reduction,
            * ``tree``   is the critical‑path length for a balanced binary‑tree reduction.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    if latency <= 0:
        raise ValueError("latency must be positive")

    serial = (n - 1) * latency
    # depth of a full binary tree that can hold n leaves is ceil(log2(n))
    import math
    tree_depth = math.ceil(math.log2(n))
    tree = tree_depth * latency
    return serial, tree
