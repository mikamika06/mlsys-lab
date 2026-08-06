import ref

def check(workdir):
    from draftopt.tree import evaluate_tree_acceptance
    traces = ref.get_traces()
    ok = 0
    for trace in traces:
        want = evaluate_tree_acceptance_ref(trace, trace["branch_factor"])
        got = evaluate_tree_acceptance(trace, trace["branch_factor"])
        if got == want:
            ok += 1
    out = {"tree_stats_matched": 1.0 if ok == len(traces) else 0.0}
    return out

def evaluate_tree_acceptance_ref(trace, branching_factor):
    depth = trace.get("depth", 3)
    total_nodes = sum(branching_factor**i for i in range(1, depth + 1))
    accepted_count = 0
    for path in trace.get("paths", []):
        valid = True
        for token in path:
            if token == 0:
                valid = False
                break
        if valid:
            accepted_count += len(path)
    single_chain_accepted = sum(p.count(1) for p in trace.get("paths", [])) / max(1, len(trace.get("paths", [])))
    return {
        "tree_accepted_tokens": float(accepted_count),
        "single_chain_accepted_tokens": float(single_chain_accepted),
        "total_nodes": float(total_nodes)
    }
