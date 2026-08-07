def select_fp16_layers(errors: list[float], k: int) -> list[int]:
    """
    Return the indices of the top‑k layers with largest FP8‑KV error.

    Parameters
    ----------
    errors : list[float]
        1‑D list of per‑layer errors.
    k : int
        Number of layers to keep in FP16 (0 ≤ k ≤ len(errors)).

    Returns
    -------
    list[int]
        Indices sorted by decreasing error; ties broken by increasing index.
    """
    if k == 0:
        return []
    indexed_errors = list(enumerate(errors))
    indexed_errors.sort(key=lambda x: (-x[1], x[0]))
    return [idx for idx, _ in indexed_errors[:k]]
