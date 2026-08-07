def decode_throughput_ratio(b1_tokens_per_sec, b64_tokens_per_sec):
    if b1_tokens_per_sec <= 0:
        return 0.0
    return float(b64_tokens_per_sec / b1_tokens_per_sec)
