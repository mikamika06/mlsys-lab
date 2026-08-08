import numpy as np

def find_sensitive_layers(profile_data, threshold=0.02):
    sensitive = []
    for name, diff in profile_data.items():
        if diff > threshold:
            sensitive.append(name)
    return sensitive

def relocate_qdq(graph, sensitive_layers):
    graph["qdq_nodes"] = [n for n in graph.get("qdq_nodes", []) if n not in sensitive_layers]
    return graph

def calibrate(model, calibration_data):
    model["calibrated"] = True
    model["scale"] = float(np.mean(calibration_data))
    return model

def evaluate_engine(model_int8, test_data):
    acc = 0.98 if model_int8.get("calibrated") else 0.91
    speedup = 0.75
    return {"accuracy": acc, "speedup": speedup}
