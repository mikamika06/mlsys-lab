def compute_node_coverage(graph_nodes, trt_nodes):
    if not graph_nodes:
        return 0.0
    covered = len(set(graph_nodes).intersection(set(trt_nodes)))
    return float(covered) / float(len(set(graph_nodes)))
