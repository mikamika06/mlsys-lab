def parse_nodes(graph_data):
    nodes = graph_data.get("nodes", [])
    result = []
    for node in nodes:
        result.append({
            "name": node.get("name"),
            "op": node.get("op"),
            "target": node.get("target"),
            "inputs": list(node.get("inputs", [])),
            "outputs": list(node.get("outputs", []))
        })
    return result
