import ref


def check(workdir):
    from trtcache.partition import compute_node_coverage, filter_subgraphs
    from trtcache.monitor import evaluate_latency_ratio

    out = {"latency_ratio": 999.0, "coverage_met": 0.0}

    cov = compute_node_coverage(ref.GRAPH_NODES, ref.SUBGRAPHS)
    filtered = filter_subgraphs(ref.SUBGRAPHS, min_nodes=2)

    cold_lat = [120.0, 118.0, 122.0]
    warm_lat = [12.0, 11.0, 13.0]
    ratio = evaluate_latency_ratio(cold_lat, warm_lat)

    out["latency_ratio"] = float(ratio)
    if cov >= 0.8 and len(filtered) == 2:
        out["coverage_met"] = 1.0
    else:
        out["_note"] = f"coverage={cov}, filtered_len={len(filtered)}"
    return out
