def eliminate_dead_code(graph_ir):
    """Remove nodes that do not reach any graph output unless marked as side-effecting."""
    nodes = graph_ir.get("nodes", [])
    outputs = set(graph_ir.get("outputs", []))

    used = set(outputs)
    for node in nodes:
        if node.get("side_effect", False):
            used.add(node["name"])

    changed = True
    while changed:
        changed = False
        for node in nodes:
            if node["name"] in used:
                for arg in node.get("args", []):
                    if isinstance(arg, str) and arg.startswith("%"):
                        ref_name = arg[1:]
                        if ref_name not in used:
                            used.add(ref_name)
                            changed = True
                for v in node.get("kwargs", {}).values():
                    if isinstance(v, str) and v.startswith("%"):
                        ref_name = v[1:]
                        if ref_name not in used:
                            used.add(ref_name)
                            changed = True

    new_nodes = [node for node in nodes if node["name"] in used]
    return {
        "inputs": list(graph_ir.get("inputs", [])),
        "outputs": list(graph_ir.get("outputs", [])),
        "nodes": new_nodes,
    }


def extract_subgraph(graph_ir, target_nodes):
    """Extract a topological subgraph containing target_nodes and their required inputs."""
    nodes_by_name = {n["name"]: n for n in graph_ir.get("nodes", [])}
    needed = set(target_nodes)

    changed = True
    while changed:
        changed = False
        for name in list(needed):
            if name in nodes_by_name:
                node = nodes_by_name[name]
                for arg in node.get("args", []):
                    if isinstance(arg, str) and arg.startswith("%"):
                        ref_name = arg[1:]
                        if ref_name not in needed and ref_name in nodes_by_name:
                            needed.add(ref_name)
                            changed = True
                for v in node.get("kwargs", {}).values():
                    if isinstance(v, str) and v.startswith("%"):
                        ref_name = v[1:]
                        if ref_name not in needed and ref_name in nodes_by_name:
                            needed.add(ref_name)
                            changed = True

    sub_nodes = [n for n in graph_ir.get("nodes", []) if n["name"] in needed]
    sub_inputs = [inp for inp in graph_ir.get("inputs", [])]
    return {
        "inputs": sub_inputs,
        "outputs": list(target_nodes),
        "nodes": sub_nodes,
    }
