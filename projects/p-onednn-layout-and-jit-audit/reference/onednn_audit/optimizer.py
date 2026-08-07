from onednn_audit.graph import OpNode, ExecutionGraph


def optimize_layer_sequence(graph):
    new_nodes = []
    i = 0
    n = len(graph.nodes)
    while i < n:
        curr = graph.nodes[i]
        if curr.is_reorder():
            if i > 0 and i < n - 1:
                prev_node = new_nodes[-1] if len(new_nodes) > 0 else graph.nodes[i - 1]
                next_node = graph.nodes[i + 1]
                if prev_node.out_layout == next_node.in_layout:
                    i += 1
                    continue
        new_nodes.append(OpNode(
            op_id=len(new_nodes),
            prim_kind=curr.prim_kind,
            in_layout=curr.in_layout,
            out_layout=curr.out_layout,
            exec_time_ms=curr.exec_time_ms
        ))
        i += 1
    for idx, node in enumerate(new_nodes):
        node.op_id = idx
    return ExecutionGraph(new_nodes)


def calculate_reorder_ratio(graph):
    total = graph.total_execution_time_ms()
    if total == 0.0:
        return 0.0
    return graph.total_reorder_time_ms() / total
