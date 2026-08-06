def reconstruct_signature(exported_program_or_graph):
    if hasattr(exported_program_or_graph, "graph_module"):
        gm = exported_program_or_graph.graph_module
        inputs = [node.name for node in gm.graph.nodes if node.op == "placeholder"]
        return {"inputs": inputs}
    elif isinstance(exported_program_or_graph, dict):
        return exported_program_or_graph.get("signature", {"inputs": []})
    return {"inputs": []}
