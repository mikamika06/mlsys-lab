import onnx


def extract_subgraph(model, input_names, output_names):
    graph = model.graph
    all_tensors = {init.name for init in graph.initializer}
    for inp in graph.input:
        all_tensors.add(inp.name)
    for node in graph.node:
        for out in node.output:
            all_tensors.add(out)

    visited_nodes = set()
    queue = list(output_names)
    target_inputs = set(input_names)

    node_by_output = {}
    for node in graph.node:
        for out in node.output:
            node_by_output[out] = node

    while queue:
        curr = queue.pop(0)
        if curr in target_inputs:
            continue
        if curr in node_by_output:
            node = node_by_output[curr]
            if node.name not in visited_nodes:
                visited_nodes.add(node.name)
                for inp in node.input:
                    if inp and inp not in target_inputs:
                        queue.append(inp)

    new_nodes = [n for n in graph.node if n.name in visited_nodes or any(o in output_names for o in n.output)]

    new_graph = onnx.helper.make_graph(
        new_nodes,
        graph.name,
        [i for i in graph.input if i.name in input_names or any(i.name in n.input for n in new_nodes)],
        [o for o in graph.output if o.name in output_names or any(o.name in n.output for n in new_nodes)],
        initializer=[init for init in graph.initializer if any(init.name in n.input for n in new_nodes)]
    )
    new_model = onnx.helper.make_model(new_graph, producer_name="mlsys-lab")
    return new_model
