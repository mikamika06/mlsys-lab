import sys
import numpy as np


def _oracle(adjacency, payloads):
    nodes = []
    for payload in payloads:
        nodes.append({"payload": list(payload), "children": []})

    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if adjacency[i, j] != 0:
                nodes[i]["children"].append(nodes[j])

    seen = set()

    def deep_size(obj):
        oid = id(obj)
        if oid in seen:
            return 0
        seen.add(oid)
        total = sys.getsizeof(obj)
        if isinstance(obj, dict):
            for k, v in obj.items():
                total += deep_size(k)
                total += deep_size(v)
        elif isinstance(obj, (list, tuple, set)):
            for item in obj:
                total += deep_size(item)
        return total

    total = 0
    for node in nodes:
        total += deep_size(node)

    flat = np.asarray([x for p in payloads for x in p], dtype=np.int64)
    return float(total / flat.nbytes)


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([[0, 1], [0, 0]], dtype=np.int64),
            [[1, 2], [3]],
        ),
        (
            np.array(
                [
                    [0, 1, 1],
                    [0, 0, 1],
                    [0, 0, 0],
                ],
                dtype=np.int64,
            ),
            [[5], [6, 7], [8, 9, 10]],
        ),
        (
            np.array(
                [
                    [0, 1, 0, 0],
                    [0, 0, 1, 1],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
                dtype=np.int64,
            ),
            [[1, 1], [2], [3, 4], [5]],
        ),
    ]

    errors = []
    for adjacency, payloads in cases:
        ref = _oracle(adjacency, payloads)
        try:
            got = float(sol.aggregate_footprint(adjacency, payloads))
        except Exception:
            return {"size_ratio": 0.0}
        errors.append(abs(got - ref) / (abs(ref) + 1e-12))

    return {"size_ratio": float(max(0.0, 1.0 - max(errors)))}
