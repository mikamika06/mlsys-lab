def get_toy_nodes():
    return 10

def get_pruned_indices():
    return [2, 5]

def get_toy_reference(nodes, indices, ratio):
    active = set(range(nodes))
    for idx in indices:
        if idx in active:
            active.remove(idx)
    return sorted(list(active))

def get_model_profile():
    return {
        "layers": [
            {"name": "conv1", "out_channels": 64},
            {"name": "conv2", "out_channels": 128}
        ]
    }

def get_dense_profile():
    return {"latency_ms": 120.0}

def get_pruned_profile():
    return {"latency_ms": 90.0, "sparsity_ratio": 0.5}
