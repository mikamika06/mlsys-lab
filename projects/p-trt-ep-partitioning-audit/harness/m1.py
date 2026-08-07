import sys


def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    import ref
    import trtep.audit as audit

    m = {"partition_api_ok": 0.0, "subgraph_count_ok": 0.0, "all_nodes_assigned": 0.0}
    try:
        g = ref.build_benchmark_graph()
        subgraphs = audit.partition_graph(g, ref.DEFAULT_SUPPORTED_OPS)
    except Exception:
        return m

    if not isinstance(subgraphs, list) or len(subgraphs) == 0:
        return m

    for sub in subgraphs:
        if not hasattr(sub, "provider") or not hasattr(sub, "nodes") or not hasattr(sub, "inputs") or not hasattr(sub, "outputs"):
            return m

    m["partition_api_ok"] = 1.0

    ref_subs = ref.ref_partition_graph(g, ref.DEFAULT_SUPPORTED_OPS)
    if len(subgraphs) == len(ref_subs):
        m["subgraph_count_ok"] = 1.0

    assigned_node_ids = []
    for sub in subgraphs:
        for n in sub.nodes:
            assigned_node_ids.append(n.node_id)

    original_ids = [n.node_id for n in g.nodes]
    if sorted(assigned_node_ids) == sorted(original_ids) and len(assigned_node_ids) == len(original_ids):
        m["all_nodes_assigned"] = 1.0

    return m
