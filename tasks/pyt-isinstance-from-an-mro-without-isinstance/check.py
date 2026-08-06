import numpy as np


def _oracle(mro_adj, pairs):
    n = len(mro_adj)
    reach = [[False] * n for _ in range(n)]
    for i in range(n):
        stack = [i]
        seen = set()
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            reach[i][node] = True
            for nxt in range(n):
                if mro_adj[node][nxt] != 0 and nxt not in seen:
                    stack.append(nxt)
    return [reach[int(a)][int(b)] for a, b in pairs]


def grade(sol, fx) -> dict:
    cases = [
        (
            [
                [1, 1, 0, 0],
                [0, 1, 1, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            [[0, 2], [1, 2], [2, 0], [3, 3]],
        ),
        (
            [
                [1, 0, 1, 0, 0],
                [0, 1, 0, 1, 0],
                [0, 0, 1, 1, 0],
                [0, 0, 0, 1, 1],
                [0, 0, 0, 0, 1],
            ],
            [[0, 3], [0, 4], [1, 4], [4, 0], [3, 3]],
        ),
        (
            [
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 1],
            ],
            [[1, 0], [2, 0], [0, 1], [2, 2]],
        ),
    ]

    ok = 1.0
    for adj, pairs in cases:
        try:
            got = sol.mro_isinstance(adj, pairs)
        except Exception:
            ok = 0.0
            break
        expected = _oracle(adj, pairs)
        if got != expected:
            ok = 0.0
            break
    return {"exact_match": ok}
