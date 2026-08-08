def get_node_counts(model_name):
    table = {
        "model_a": {"relay_nodes": 12, "relax_nodes": 9},
        "model_b": {"relay_nodes": 15, "relax_nodes": 11},
        "model_c": {"relay_nodes": 18, "relax_nodes": 14},
    }
    return table.get(model_name)
