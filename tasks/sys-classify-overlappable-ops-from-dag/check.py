import numpy as np


def _descendants(n, edges, u):
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
    seen = set()
    stack = [u]
    while stack:
        x = stack.pop()
        for y in adj[x]:
            if y not in seen:
                seen.add(y)
                stack.append(y)
    return seen


def _oracle(types, edges):
    n = len(types)
    desc = [_descendants(n, edges, u) for u in range(n)]
    out = set()
    for c in range(n):
        if types[c] != "comm":
            continue
        for k in range(n):
            if types[k] != "compute":
                continue
            if k not in desc[c] and c not in desc[k]:
                out.add((c, k))
    return out


def _gen_case(rng):
    n = int(rng.integers(6, 17))
    types = ["comm" if rng.random() < (1.0 / 3.0) else "compute" for _ in range(n)]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.15:
                edges.append((i, j))
    return types, edges


def grade(sol, fx) -> dict:
    cases = []
    # fixed example from task.md
    cases.append((
        ["compute", "comm", "compute", "comm", "compute"],
        [(0, 1), (1, 4), (2, 3)],
    ))
    rng = np.random.default_rng(0)
    for _ in range(8):
        cases.append(_gen_case(rng))

    ok = 1.0
    for types, edges in cases:
        expected = _oracle(types, edges)
        try:
            got = sol.overlappable_ops(list(types), list(edges))
            got_norm = {(int(a), int(b)) for a, b in got}
        except Exception:
            ok = 0.0
            break
        if got_norm != expected:
            ok = 0.0
            break
    return {"exact_match": ok}
