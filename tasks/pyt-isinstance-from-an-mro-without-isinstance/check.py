import numpy as np


def _oracle(mro_adj, pairs):
    n = mro_adj.shape[0]
    reach = np.zeros((n, n), dtype=bool)
    for i in range(n):
        stack = [i]
        seen = set()
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            reach[i, node] = True
            for nxt in range(n):
                if mro_adj[node, nxt] != 0 and nxt not in seen:
                    stack.append(nxt)
    return np.array([reach[int(a), int(b)] for a, b in pairs], dtype=bool)


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([
                [1, 1, 0, 0],
                [0, 1, 1, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ], dtype=int),
            np.array([[0, 2], [1, 2], [2, 0], [3, 3]]),
        ),
        (
            np.array([
                [1, 0, 1, 0, 0],
                [0, 1, 0, 1, 0],
                [0, 0, 1, 1, 0],
                [0, 0, 0, 1, 1],
                [0, 0, 0, 0, 1],
            ], dtype=int),
            np.array([[0, 3], [0, 4], [1, 4], [4, 0], [3, 3]]),
        ),
        (
            np.array([
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 1],
            ], dtype=int),
            np.array([[1, 0], [2, 0], [0, 1], [2, 2]]),
        ),
    ]

    ok = 1.0
    for adj, pairs in cases:
        try:
            got = np.asarray(sol.mro_isinstance(adj, pairs), dtype=bool)
        except Exception:
            ok = 0.0
            break
        expected = _oracle(adj, pairs)
        if got.shape != expected.shape or not np.array_equal(got, expected):
            ok = 0.0
            break
    return {"exact_match": ok}
