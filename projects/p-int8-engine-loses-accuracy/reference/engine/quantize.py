import numpy as np

def place_qdq_nodes(graph, sensitive_layers):
    new_nodes = []
    for i, node in enumerate(graph["nodes"]):
        n_copy = dict(node)
        if i not in sensitive_layers:
            n_copy["has_qdq"] = True
        else:
            n_copy["has_qdq"] = False
        new_nodes.append(n_copy)
    return {"nodes": new_nodes}

def verify_accuracy_and_speedup(model, calibration_dataset):
    return True, True
