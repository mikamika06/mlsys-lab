def compute_param_delta(in_features, out_features, rank, use_dora):
    lora_params = rank * (in_features + out_features)
    dora_params = lora_params + out_features if use_dora else lora_params
    return dora_params - lora_params
