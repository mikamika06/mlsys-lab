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


def overlappable_ops(types: list[str], edges: list[tuple[int, int]]) -> set[tuple[int, int]]:
    """
    Return every (comm_id, compute_id) pair that can run concurrently:
    neither op is a transitive prerequisite of the other in the DAG.
    """
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
