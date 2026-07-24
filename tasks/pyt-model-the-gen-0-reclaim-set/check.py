def _oracle(nodes, edges, refcounts, roots):
    nodes = set(nodes)

    internal_in = {n: 0 for n in nodes}
    for src, dsts in edges.items():
        if src in nodes:
            for dst in dsts:
                if dst in nodes:
                    internal_in[dst] += 1

    trial = {}
    for n in nodes:
        trial[n] = refcounts[n] - internal_in[n]

    reachable = set(roots)
    stack = list(roots)
    while stack:
        cur = stack.pop()
        for nxt in edges.get(cur, []):
            if nxt in nodes and nxt not in reachable:
                reachable.add(nxt)
                stack.append(nxt)

    positive = {n for n in nodes if trial[n] > 0}
    keep = set(positive) | reachable

    changed = True
    while changed:
        changed = False
        for n in nodes - keep:
            if any(dst in keep for dst in edges.get(n, [])):
                keep.add(n)
                changed = True

    return nodes - keep


def grade(sol, fx) -> dict:
    cases = [
        (
            [1, 2, 3, 4],
            {1: [2], 2: [1], 3: [4], 4: []},
            {1: 1, 2: 1, 3: 1, 4: 0},
            [3],
        ),
        (
            [10, 11, 12],
            {10: [11], 11: [12], 12: [10]},
            {10: 1, 11: 1, 12: 1},
            [],
        ),
        (
            [5, 6, 7, 8],
            {5: [6], 6: [5, 7], 7: [], 8: [7]},
            {5: 1, 6: 2, 7: 2, 8: 1},
            [8],
        ),
        (
            [20, 21, 22, 23],
            {20: [21], 21: [20], 22: [23], 23: [22]},
            {20: 1, 21: 1, 22: 1, 23: 1},
            [20],
        ),
    ]

    ok = 1.0
    for nodes, edges, refs, roots in cases:
        try:
            got = set(sol.reclaim_set(nodes, edges, refs, roots))
        except Exception:
            ok = 0.0
            break
        if got != _oracle(nodes, edges, refs, roots):
            ok = 0.0
            break

    return {"exact_match": ok}
