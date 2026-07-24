def reclaim_set(nodes, edges, refcounts, roots):
    node_set = set(nodes)

    internal_in = {n: 0 for n in node_set}
    for src, targets in edges.items():
        if src in node_set:
            for target in targets:
                if target in node_set:
                    internal_in[target] += 1

    trial = {
        n: refcounts[n] - internal_in[n]
        for n in node_set
    }

    reachable = set(roots)
    stack = list(roots)
    while stack:
        node = stack.pop()
        for target in edges.get(node, []):
            if target in node_set and target not in reachable:
                reachable.add(target)
                stack.append(target)

    keep = set(reachable)
    keep.update(n for n in node_set if trial[n] > 0)

    changed = True
    while changed:
        changed = False
        for node in node_set - keep:
            for target in edges.get(node, []):
                if target in keep:
                    keep.add(node)
                    changed = True
                    break

    return node_set - keep
