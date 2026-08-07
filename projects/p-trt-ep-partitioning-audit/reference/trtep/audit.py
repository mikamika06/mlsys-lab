class Node:
    def __init__(self, node_id, op_type, inputs, outputs, attrs=None):
        self.node_id = node_id
        self.op_type = op_type
        self.inputs = list(inputs)
        self.outputs = list(outputs)
        self.attrs = attrs or {}


class Graph:
    def __init__(self, nodes, inputs, outputs):
        self.nodes = list(nodes)
        self.inputs = list(inputs)
        self.outputs = list(outputs)


class Subgraph:
    def __init__(self, sub_id, provider, nodes, inputs, outputs):
        self.sub_id = sub_id
        self.provider = provider
        self.nodes = list(nodes)
        self.inputs = list(inputs)
        self.outputs = list(outputs)


def partition_graph(graph, supported_ops):
    subgraphs = []
    current_nodes = []
    current_provider = None
    sub_counter = 0

    for node in graph.nodes:
        provider = "TensorRT" if node.op_type in supported_ops else "CPU"
        if current_provider is None:
            current_provider = provider
            current_nodes = [node]
        elif provider == current_provider:
            current_nodes.append(node)
        else:
            subgraphs.append(_create_subgraph(sub_counter, current_provider, current_nodes, graph))
            sub_counter += 1
            current_provider = provider
            current_nodes = [node]

    if current_nodes:
        subgraphs.append(_create_subgraph(sub_counter, current_provider, current_nodes, graph))

    return subgraphs


def _create_subgraph(sub_id, provider, nodes, graph):
    produced_tensors = set()
    for n in nodes:
        produced_tensors.update(n.outputs)

    consumed_tensors = set()
    for n in nodes:
        consumed_tensors.update(n.inputs)

    graph_inputs = set(graph.inputs)
    sub_inputs = [t for t in consumed_tensors if t not in produced_tensors or t in graph_inputs]

    sub_outputs = set()
    for n in nodes:
        for out_t in n.outputs:
            if out_t in graph.outputs:
                sub_outputs.add(out_t)
            else:
                for other_node in graph.nodes:
                    if other_node not in nodes and out_t in other_node.inputs:
                        sub_outputs.add(out_t)
                        break

    return Subgraph(sub_id, provider, nodes, sorted(list(sub_inputs)), sorted(list(sub_outputs)))


def find_partition_breakers(graph, supported_ops):
    breakers = []
    nodes = graph.nodes
    n_count = len(nodes)

    for i, node in enumerate(nodes):
        if node.op_type not in supported_ops:
            has_prev_trt = any(nodes[j].op_type in supported_ops for j in range(i))
            has_next_trt = any(nodes[j].op_type in supported_ops for j in range(i + 1, n_count))
            if has_prev_trt and has_next_trt:
                if node.op_type not in breakers:
                    breakers.append(node.op_type)

    return breakers
