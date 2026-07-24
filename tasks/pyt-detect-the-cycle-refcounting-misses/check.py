def _oracle(graph):
    nodes = list(graph.keys())
    index = 0
    stack = []
    on_stack = set()
    indices = {}
    low = {}
    comps = []

    def visit(v):
        nonlocal index
        indices[v] = index
        low[v] = index
        index += 1
        stack.append(v)
        on_stack.add(v)

        for w in graph[v]:
            if w not in indices:
                visit(w)
                low[v] = min(low[v], low[w])
            elif w in on_stack:
                low[v] = min(low[v], indices[w])

        if low[v] == indices[v]:
            comp = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                comp.append(w)
                if w == v:
                    break
            comps.append(comp)

    for n in nodes:
        if n not in indices:
            visit(n)

    result = []
    for comp in comps:
        s = set(comp)
        cyclic = len(s) > 1 or any(x in graph[x] for x in s)
        if not cyclic:
            continue
        incoming = False
        for src, dsts in graph.items():
            if src not in s and any(dst in s for dst in dsts):
                incoming = True
                break
        if not incoming:
            result.append(sorted(s))
    return sorted(result, key=lambda x: x[0])


def grade(sol, fx) -> dict:
    cases = [
        {
            1: [2],
            2: [1],
            3: [],
        },
        {
            1: [1],
            2: [1],
            3: [2],
        },
        {
            1: [2],
            2: [3],
            3: [1],
            4: [3],
        },
        {
            1: [2],
            2: [],
            3: [4],
            4: [5],
            5: [3],
            6: [],
        },
        {
            10: [11],
            11: [12],
            12: [10, 13],
            13: [14],
            14: [13],
            15: [14],
        },
    ]

    ok = 1.0
    for graph in cases:
        try:
            got = sol.uncollectable_cycles({k: list(v) for k, v in graph.items()})
        except Exception:
            ok = 0.0
            break
        expected = _oracle(graph)
        if got != expected:
            ok = 0.0
            break
    return {"exact_match": ok}
