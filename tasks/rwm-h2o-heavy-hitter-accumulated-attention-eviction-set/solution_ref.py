import math


def h2o_eviction_set(attn_scores: list[list[float]], budget: int, recent_window: int):
    """
    H2O (Heavy-Hitter Oracle) static eviction set.

    attn_scores : (n, n) raw attention logits.
    budget      : number of tokens to keep (recent_window <= budget <= n).
    recent_window : number of most-recent positions always kept.

    Returns (retained_idx, preserved_mass):
      retained_idx   : list of int, ascending, length == budget.
      preserved_mass : float, fraction of total accumulated attention mass
                        captured by the retained set.
    """
    S = attn_scores
    n = len(S)

    P = [[0.0] * n for _ in range(n)]

    for i in range(n):
        row_max = S[i][0]
        for j in range(1, i + 1):
            if S[i][j] > row_max:
                row_max = S[i][j]

        row_sum = 0.0
        row_exps = [0.0] * (i + 1)
        for j in range(i + 1):
            val = math.exp(S[i][j] - row_max)
            row_exps[j] = val
            row_sum += val

        for j in range(i + 1):
            P[i][j] = row_exps[j] / row_sum

    h = [0.0] * n
    for j in range(n):
        col_sum = 0.0
        for i in range(n):
            col_sum += P[i][j]
        h[j] = col_sum

    start_recent = n - recent_window
    if start_recent < 0:
        start_recent = 0

    recent_set = set(range(start_recent, n))
    n_heavy = budget - len(recent_set)

    candidates = [j for j in range(n) if j not in recent_set]

    cand_len = len(candidates)
    for i in range(1, cand_len):
        key_j = candidates[i]
        key_h = h[key_j]
        j = i - 1
        while j >= 0:
            cj = candidates[j]
            hj = h[cj]
            if (key_h > hj) or (key_h == hj and key_j < cj):
                candidates[j + 1] = candidates[j]
                j -= 1
            else:
                break
        candidates[j + 1] = key_j

    heavy = candidates[:n_heavy]

    retained_set = set(heavy)
    for j in range(start_recent, n):
        retained_set.add(j)

    retained = list(retained_set)
    ret_len = len(retained)
    for i in range(1, ret_len):
        key = retained[i]
        j = i - 1
        while j >= 0 and retained[j] > key:
            retained[j + 1] = retained[j]
            j -= 1
        retained[j + 1] = key

    retained_idx = retained

    retained_mass = 0.0
    for idx in retained:
        retained_mass += h[idx]

    total_h = 0.0
    for j in range(n):
        total_h += h[j]

    preserved_mass = float(retained_mass / total_h)
    return retained_idx, preserved_mass
