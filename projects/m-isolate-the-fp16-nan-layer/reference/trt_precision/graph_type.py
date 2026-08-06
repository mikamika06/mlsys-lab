def convert_to_weak_typed_fp16(graph):
    nodes = []
    for node in graph["nodes"]:
        n = dict(node)
        n["dtype"] = "FLOAT16"
        n["precision_mode"] = "weak_typed"
        nodes.append(n)
    return {"nodes": nodes, "mode": "weak_typed"}
