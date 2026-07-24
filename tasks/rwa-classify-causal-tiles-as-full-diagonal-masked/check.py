import numpy as np


def _ref_classify(seq_len, block_size):
    num_blocks = seq_len // block_size
    result = []
    for i in range(num_blocks):
        row = []
        for j in range(num_blocks):
            q_start = i * block_size
            q_end = (i + 1) * block_size
            k_start = j * block_size
            k_end = (j + 1) * block_size
            # Vectorized oracle: k <= q for all pairs
            qs = np.arange(q_start, q_end)
            ks = np.arange(k_start, k_end)
            mask = ks[None, :] <= qs[:, None]
            if mask.all():
                row.append("full")
            elif not mask.any():
                row.append("empty")
            else:
                row.append("diagonal")
        result.append(row)
    return result


def grade(sol, fx) -> dict:
    test_cases = [
        (4, 2),
        (8, 2),
        (8, 4),
        (16, 4),
        (12, 3),
        (16, 8),
    ]
    ok = 1.0
    for seq_len, block_size in test_cases:
        ref = _ref_classify(seq_len, block_size)
        try:
            got = sol.classify_causal_tiles(seq_len, block_size)
            got = [list(row) for row in got]
        except Exception:
            ok = 0.0
            break
        if got != ref:
            ok = 0.0
            break
    return {"exact_match": ok}
