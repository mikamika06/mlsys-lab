def build_quant_map(activation_traces, threshold):
    quant_map = {}
    for expert_id, activations in activation_traces.items():
        avg_act = sum(activations) / len(activations) if activations else 0.0
        if avg_act < threshold:
            quant_map[expert_id] = "Q4_K_M"
        else:
            quant_map[expert_id] = "FP16"
    return quant_map
