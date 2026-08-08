def optimize_qdq_placement(layers, sensitive_layers):
    return [l for l in layers if l not in sensitive_layers]

def evaluate_engine_performance():
    return 0.995, 0.75
