from ortopt.graph import Node, Graph

def apply_attention_fusion(graph):
    new_nodes = []
    skip = 0
    for i in range(len(graph.nodes)):
        if skip > 0:
            skip -= 1
            continue
        node = graph.nodes[i]
        if node.op_type == "MatMul" and i + 3 < len(graph.nodes):
            if (graph.nodes[i+1].op_type == "MatMul" and
                graph.nodes[i+2].op_type == "MatMul" and
                graph.nodes[i+3].op_type == "Softmax"):
                fused = Node("fused_attn", "FusedAttention", [node.inputs[0], graph.nodes[i+1].inputs[0], graph.nodes[i+2].inputs[0]], [graph.nodes[i+3].outputs[0]])
                new_nodes.append(fused)
                skip = 3
                continue
        new_nodes.append(node)
    return Graph(new_nodes)
