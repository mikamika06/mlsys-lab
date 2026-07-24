def dead_code_elimination(
    nodes,
    outputs
):
    """Remove all nodes that cannot reach any of the given output ids."""
    node_map = {n["id"]: n for n in nodes}
    reachable = set()
    stack = list(outputs)
    while stack:
        nid = stack.pop()
        if nid not in node_map or nid in reachable:
            continue
        reachable.add(nid)
        stack.extend(node_map[nid].get("inputs", []))
    new_nodes = [node_map[nid] for nid in sorted(reachable) if nid in node_map]
    new_outputs = sorted([o for o in outputs if o in reachable])
    return new_nodes, new_outputs
