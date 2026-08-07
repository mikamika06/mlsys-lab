from trtep.audit import Graph, Node

OP_REPLACEMENTS = {
    "CustomGelu": "Gelu",
    "UnsupportedPad": "Pad",
    "LegacyNorm": "LayerNormalization"
}


def rewrite_graph(graph, supported_ops):
    new_nodes = []
    for node in graph.nodes:
        op_type = node.op_type
        if op_type in OP_REPLACEMENTS:
            replacement = OP_REPLACEMENTS[op_type]
            if replacement in supported_ops:
                op_type = replacement
        new_node = Node(node.node_id, op_type, node.inputs, node.outputs, dict(node.attrs))
        new_nodes.append(new_node)
    return Graph(new_nodes, list(graph.inputs), list(graph.outputs))
