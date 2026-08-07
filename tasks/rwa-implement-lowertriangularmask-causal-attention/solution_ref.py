def causal_mask(logits: list[list[float]]) -> list[list[float]]:
    """
    Return logits with -inf added to all strictly upper-triangular entries.
    The output is a 2D list of floats of the same shape as ``logits``.
    """
    out = []
    for i, row in enumerate(logits):
        new_row = []
        for j, val in enumerate(row):
            if i < j:
                new_row.append(float("-inf"))
            else:
                new_row.append(float(val))
        out.append(new_row)
    return out
