def _reference(nodes, edges):
    children = {node: [] for node in nodes}
    indegree = {node: 0 for node in nodes}
    for src, dst in edges:
        children[src].append(dst)
        indegree[dst] += 1

    queue = [node for node in nodes if indegree[node] == 0]
    finish = {node: 0 for node in nodes}

    while queue:
        node = queue.pop(0)
        finish[node] += nodes[node]
        for child in children[node]:
            if finish[child] < finish[node]:
                finish[child] = finish[node]
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)

    return max(finish.values(), default=0)


def grade(sol, fx) -> dict:
    cases = [
        (
            {0: 3, 1: 5, 2: 2, 3: 4},
            [(0, 1), (0, 2), (1, 3), (2, 3)],
        ),
        (
            {0: 7, 1: 1, 2: 6, 3: 8, 4: 2},
            [(0, 1), (0, 2), (2, 3), (1, 4), (3, 4)],
        ),
        (
            {0: 4, 1: 9, 2: 3, 3: 2},
            [],
        ),
        (
            {0: 2, 1: 2, 2: 2, 3: 2, 4: 10},
            [(0, 2), (1, 2), (2, 3), (3, 4)],
        ),
    ]

    ok = 1.0
    for nodes, edges in cases:
        try:
            got = sol.critical_path_length(dict(nodes), list(edges))
        except Exception:
            ok = 0.0
            break
        if got != _reference(dict(nodes), list(edges)):
            ok = 0.0
            break

    return {"exact_match": ok}
