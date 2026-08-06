def count_parameters(in_features, out_features, r):
    base_weight = out_features * in_features
    base_bias = out_features
    lora_a = r * in_features
    lora_b = out_features * r
    return {
        "base_weight": base_weight,
        "base_bias": base_bias,
        "lora_a": lora_a,
        "lora_b": lora_b,
        "total_lora": lora_a + lora_b,
        "total": base_weight + base_bias + lora_a + lora_b
    }
