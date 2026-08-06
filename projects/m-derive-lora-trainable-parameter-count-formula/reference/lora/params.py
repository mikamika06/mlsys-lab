def lora_params_count(in_features, out_features, rank):
    return int(rank * in_features + out_features * rank)
