def optimize_graph(parsed_graph):
    nodes = parsed_graph.get("nodes", [])
    filtered = [n for n in nodes if n.get("op") != "noop"]
    return {"nodes": filtered, "metadata": parsed_graph.get("metadata", {})}
