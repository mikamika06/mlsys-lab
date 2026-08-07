def count_live_params_and_sparsity(mask: list[list[int]]) -> tuple[int, float]:
    """
    Count the number of non‑zero entries in a mask and compute its sparsity.

    Parameters
    ----------
    mask : list[list[int]]
        2‑D list containing zeros and ones (or any truthy values).

    Returns
    -------
    live_count : int
        Number of non‑zero elements.
    sparsity : float
        Fraction of zero elements in the mask, in [0,1].
    """
    live_count = 0
    total_elements = 0
    for row in mask:
        for val in row:
            total_elements += 1
            if val:
                live_count += 1

    if total_elements == 0:
        sparsity = 0.0
    else:
        sparsity = float((total_elements - live_count) / total_elements)
    return live_count, sparsity
