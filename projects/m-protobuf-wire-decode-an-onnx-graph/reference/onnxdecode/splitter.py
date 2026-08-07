def split_weights_and_graph(data: bytes) -> tuple:
    split_idx = len(data) // 2
    graph_bytes = data[:split_idx]
    weights_bytes = data[split_idx:]
    return graph_bytes, weights_bytes
