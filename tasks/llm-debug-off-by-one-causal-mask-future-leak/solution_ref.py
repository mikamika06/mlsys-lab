def create_causal_mask(n: int) -> list[list[float]]:
    """
    Return an (n, n) causal mask with ones on and below the main diagonal.
    The result is a list of lists of floats.
    """
    mask = []
    for i in range(n):
        row = []
        for j in range(n):
            if j <= i:
                row.append(1.0)
            else:
                row.append(0.0)
        mask.append(row)
    return mask
