import numpy as np


def _oracle_rank(groups):
    scored = []
    for group in groups:
        squared_sum = 0.0
        for tensor in group["tensors"]:
            arr = np.asarray(tensor, dtype=np.float64)
            squared_sum += float(np.sum(arr * arr))
        importance = float(np.sqrt(squared_sum))
        scored.append((importance, group["id"]))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [item[1] for item in scored]


def _spearman(reference, candidate):
    if len(reference) != len(candidate):
        return 0.0
    if set(reference) != set(candidate):
        return 0.0
    if len(reference) <= 1:
        return 1.0

    ref_rank = {value: index for index, value in enumerate(reference)}
    cand_rank = {value: index for index, value in enumerate(candidate)}

    x = np.asarray([ref_rank[v] for v in reference], dtype=np.float64)
    y = np.asarray([cand_rank[v] for v in reference], dtype=np.float64)

    x -= np.mean(x)
    y -= np.mean(y)

    denominator = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denominator == 0:
        return 1.0 if np.array_equal(reference, candidate) else 0.0
    return float(np.sum(x * y) / denominator)


def grade(sol, fx) -> dict:
    cases = [
        [
            {
                "id": 10,
                "tensors": [
                    np.array([1.0, 1.0]),
                    np.array([2.0]),
                ],
            },
            {
                "id": 20,
                "tensors": [
                    np.array([3.0]),
                ],
            },
            {
                "id": 30,
                "tensors": [
                    np.array([0.5, 0.5, 0.5, 0.5]),
                    np.array([1.0]),
                ],
            },
        ],
        [
            {
                "id": 1,
                "tensors": [
                    np.array([0.2, -0.4]),
                    np.array([3.0, 4.0]),
                ],
            },
            {
                "id": 2,
                "tensors": [
                    np.array([5.0]),
                    np.array([0.1, 0.2, 0.3]),
                ],
            },
            {
                "id": 3,
                "tensors": [
                    np.array([2.0, 2.0, 2.0]),
                ],
            },
        ],
        [
            {
                "id": 100,
                "tensors": [
                    np.array([1.0]),
                    np.array([9.0]),
                ],
            },
            {
                "id": 200,
                "tensors": [
                    np.array([7.0]),
                    np.array([7.0]),
                ],
            },
            {
                "id": 300,
                "tensors": [
                    np.array([2.0, 3.0, 4.0]),
                ],
            },
        ],
    ]

    for groups in cases:
        try:
            result = list(sol.rank_groups_by_importance(groups))
        except Exception:
            return {"spearman": 0.0}

        expected = _oracle_rank(groups)
        if _spearman(expected, result) != 1.0:
            return {"spearman": 0.0}

    return {"spearman": 1.0}
