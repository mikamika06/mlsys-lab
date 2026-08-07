import numpy as np


def grade(sol, fx) -> dict:
    weights = np.array([
        [[3.0, 0.0]],
        [[1.0, 1.0]],
        [[0.5, 0.5]],
    ])
    grads = np.array([
        [[0.1, 0.1]],
        [[5.0, 5.0]],
        [[0.1, 0.1]],
    ])

    # Convert to plain lists for the learner's function
    w_list = weights.tolist()
    g_list = grads.tolist()

    try:
        mag_rank, taylor_rank = sol.rank_heads_by_importance(w_list, g_list)
    except Exception as e:
        return {"exact_match": 0, "rankings_differ": 0, "error": str(e)}

    # Oracle computation using the gate's reference (using numpy as before)
    h = weights.shape[0]
    flat_w = weights.reshape(h, -1)
    flat_g = grads.reshape(h, -1)
    d = flat_w.shape[1]

    mag_scores = np.empty(h, dtype=np.float64)
    taylor_scores = np.empty(h, dtype=np.float64)

    for i in range(h):
        sum_sq = 0.0
        sum_taylor = 0.0
        for j in range(d):
            w_val = flat_w[i, j]
            g_val = flat_g[i, j]
            sum_sq += w_val * w_val
            sum_taylor += abs(g_val * w_val)
        mag_scores[i] = np.sqrt(sum_sq)
        taylor_scores[i] = sum_taylor

    oracle_mag = sorted(range(h), key=lambda i: (-float(mag_scores[i]), i))
    oracle_taylor = sorted(range(h), key=lambda i: (-float(taylor_scores[i]), i))

    exact_match = int(
        list(mag_rank) == list(oracle_mag) and list(taylor_rank) == list(oracle_taylor)
    )
    rankings_differ = int(list(mag_rank) != list(taylor_rank))

    return {
        "exact_match": exact_match,
        "rankings_differ": rankings_differ,
    }
