def parse_graph(serialized_program):
    nodes = []
    for node in serialized_program.get("nodes", []):
        nodes.append({
            "name": node.get("name"),
            "op": node.get("op"),
            "target": node.get("target"),
            "inputs": list(node.get("inputs", [])),
            "outputs": list(node.get("outputs", []))
        })
    return {"nodes": nodes, "metadata": serialized_program.get("metadata", {})}
