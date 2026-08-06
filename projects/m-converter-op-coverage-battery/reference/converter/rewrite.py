def apply_decomposition_table(graph_spec, equivalence_table):
    new_nodes = []
    decomposed_count = 0

    for node in graph_spec.get("nodes", []):
        op_type = node["op_type"]
        if op_type in equivalence_table:
            rules = equivalence_table[op_type]
            sub_nodes = rules["decompose_fn"](node)
            new_nodes.extend(sub_nodes)
            decomposed_count += 1
        else:
            new_nodes.append(node)

    return {
        "nodes": new_nodes,
        "decomposed_count": decomposed_count
    }
