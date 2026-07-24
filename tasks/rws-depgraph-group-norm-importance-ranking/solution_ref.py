import numpy as np


def rank_groups_by_importance(groups):
    scored = []

    for group in groups:
        squared_sum = 0.0
        for tensor in group["tensors"]:
            arr = np.asarray(tensor, dtype=np.float64)
            squared_sum += np.sum(arr * arr)

        score = float(np.sqrt(squared_sum))
        scored.append((score, group["id"]))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [group_id for _, group_id in scored]
