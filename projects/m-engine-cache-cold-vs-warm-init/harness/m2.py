import ref


def check(workdir):
    from trt_engine.partition import compute_node_coverage, optimize_subgraph_partitions

    out = {"coverage_ratio": 0.0, "partition_match": 0.0, "_note": ""}
    cov = compute_node_coverage(ref.GRAPH_NODES, [{"nodes": ref.GRAPH_NODES}])
    out["coverage_ratio"] = float(cov)

    parts = optimize_subgraph_partitions(ref.GRAPH_NODES, ref.SUPPORTED_OPS)
    if len(parts) == 1 and parts[0]["nodes"] == ref.GRAPH_NODES:
        out["partition_match"] = 1.0
    else:
        out["partition_match"] = 0.0

    return out
