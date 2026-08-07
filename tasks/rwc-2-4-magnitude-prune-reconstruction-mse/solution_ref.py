def magnitude_prune_mse(W: list[list[float]]) -> float:
    """
    Compute the mean squared error between W and its 2:4 magnitude‑pruned version.
    """
    if not W:
        return 0.0

    total_sq_diff = 0.0
    total_elements = 0

    for row in W:
        n_cols = len(row)
        total_elements += n_cols
        for j in range(0, n_cols, 4):
            group = row[j:j+4]
            g_len = len(group)
            if g_len == 0:
                continue

            if g_len <= 2:
                pass
            else:
                indexed_vals = [(abs(group[k]), k) for k in range(g_len)]
                sorted_by_val = sorted(indexed_vals)
                kept_indices = set(k for val, k in sorted_by_val[-2:])
                for k in range(g_len):
                    if k not in kept_indices:
                        total_sq_diff += group[k] ** 2

    return total_sq_diff / total_elements if total_elements > 0 else 0.0
