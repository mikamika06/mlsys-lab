import ref

def check(workdir):
    from prune_net.toy_dep import propagate_toy_dependencies
    nodes = ref.get_toy_nodes()
    indices = ref.get_pruned_indices()
    ratio = 0.2
    want = ref.get_toy_reference(nodes, indices, ratio)
    got = propagate_toy_dependencies(nodes, indices, ratio)
    ok = 1.0 if got == want else 0.0
    out = {"toy_propagated_correctly": ok}
    if ok != 1.0:
        out["_note"] = f"got {got}, want {want}"
    return out
