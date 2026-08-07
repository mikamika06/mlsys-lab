def simulate_eviction(tree_state, target_id):
    nodes = {n["id"]: dict(n) for n in tree_state["nodes"]}
    refcounts = {n["id"]: n["refcount"] for n in nodes.values()}
    parents = {n["id"]: n["parent"] for n in nodes.values()}

    current = target_id
    evicted = []
    while current is not None:
        if refcounts.get(current, 0) > 1:
            refcounts[current] -= 1
            break
        refcounts[current] = 0
        evicted.append(current)
        p = parents.get(current)
        if p is not None and p in nodes:
            if current in nodes[p].get("children", []):
                nodes[p]["children"].remove(current)
        current = p
    return sorted(evicted)
