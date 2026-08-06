def evaluate_fp8_kv(fp16_bytes, fp8_bytes):
    gain = fp16_bytes / fp8_bytes
    risk = "precision loss leading to perplexity degradation or attention drift"
    return gain, risk
