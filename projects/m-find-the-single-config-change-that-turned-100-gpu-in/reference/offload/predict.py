def predict_tok_s(bandwidth_gb_s, bytes_per_token, split_ratio):
    effective_bw = bandwidth_gb_s * 1e9
    bytes_needed = bytes_per_token * (1.0 - 0.5 * split_ratio)
    return effective_bw / bytes_needed
