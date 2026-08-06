def build_hot_cold_map(activation_trace, threshold):
    mapping = {}
    for expert_id, score in activation_trace.items():
        if score >= threshold:
            mapping[expert_id] = "FP16"
        else:
            mapping[expert_id] = "Q4_K_M"
    return mapping
