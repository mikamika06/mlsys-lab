MODELS = [
    {"name": "model_a", "ops": ["add", "multiply", "relu"], "relay_nodes": 12, "relax_nodes": 9},
    {"name": "model_b", "ops": ["subtract", "divide", "exp"], "relay_nodes": 15, "relax_nodes": 11},
    {"name": "model_c", "ops": ["matmul", "add", "tanh"], "relay_nodes": 18, "relax_nodes": 14},
]


def get_reference_counts(model_name):
    for m in MODELS:
        if m["name"] == model_name:
            return {"relay_nodes": m["relay_nodes"], "relax_nodes": m["relax_nodes"]}
    return None


def get_folding_discrepancy(model_name):
    if model_name == "model_a":
        return {"relay_folded": 3.14159, "relax_folded": 3.14160, "diff": 0.00001}
    elif model_name == "model_b":
        return {"relay_folded": 2.71828, "relax_folded": 2.71830, "diff": 0.00002}
    else:
        return {"relay_folded": 1.0, "relax_folded": 1.0, "diff": 0.0}
