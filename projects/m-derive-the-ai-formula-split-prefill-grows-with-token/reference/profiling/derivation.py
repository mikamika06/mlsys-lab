def derive_ai_formula(params, hidden_size, num_layers, total_weight_bytes, prompt_tokens):
    prefill_flops_per_token = 2.0 * params
    decode_flops_per_token = 2.0 * params
    prefill_bytes_per_token = total_weight_bytes / prompt_tokens + 2 * hidden_size * num_layers
    decode_bytes_per_token = total_weight_bytes + 2 * hidden_size * num_layers * 2
    return {
        "prefill_flops_per_token": float(prefill_flops_per_token),
        "decode_flops_per_token": float(decode_flops_per_token),
        "prefill_bytes_per_token": float(prefill_bytes_per_token),
        "decode_bytes_per_token": float(decode_bytes_per_token)
    }
