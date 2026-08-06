import math
import numpy as np


def rank_groups_by_importance(groups):
    scored = []

    for group in groups:
        squared_sum = 0.0
        for tensor in group["tensors"]:
            arr = np.asarray(tensor, dtype=np.float64)
            flat = arr.ravel()
            n = flat.shape[0]
            for i in range(n):
                val = float(flat[i])
                squared_sum += val * val

        score = float(math.sqrt(squared_sum))
        scored.append((score, group["id"]))

    n_scored = len(scored)
    for i in range(n_scored):
        for j in range(0, n_scored - i - 1):
            score_a, id_a = scored[j]
            score_b, id_b = scored[j + 1]
            if (-score_a, id_a) > (-score_b, id_b):
                scored[j], scored[j + 1] = scored[j + 1], scored[j]

    return [group_id for _, group_id in scored]
