def classify_steps(steps, bytes_per_param, peak_flops, peak_bandwidth):
    ai_ridge = peak_flops / peak_bandwidth
    labels = []
    for decode_tokens, prefill_tokens in steps:
        T = decode_tokens + prefill_tokens
        ai = 2.0 * T / bytes_per_param
        labels.append("compute" if ai >= ai_ridge else "memory")
    return labels
