def count_fake_quantize_nodes(model_graph):
    count = 0
    for node in model_graph.get("nodes", []):
        if node.get("op") == "FakeQuantize":
            count += 1
    return count
