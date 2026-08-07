def select_top2_mask(weights: list[float]) -> list[bool]:
    """Select top 2 elements by absolute value in each group of 4."""
    mask = [False] * len(weights)
    n = len(weights)

    for i in range(0, n, 4):
        vals = [0.0, 0.0, 0.0, 0.0]
        for j in range(4):
            w = weights[i + j]
            vals[j] = -w if w < 0 else w

        idxs = [0, 1, 2, 3]
        for j in range(1, 4):
            key_idx = idxs[j]
            key_val = vals[key_idx]
            k = j - 1
            while k >= 0 and vals[idxs[k]] < key_val:
                idxs[k + 1] = idxs[k]
                k -= 1
            idxs[k + 1] = key_idx

        mask[i + idxs[0]] = True
        mask[i + idxs[1]] = True

    return mask
