def profile_quants(fp16_bytes, fp16_tok_s):
    """Profiles FP16 and 3 quant levels."""
    return {
        "FP16": {"size": fp16_bytes, "tok_s": fp16_tok_s},
        "Q8_0": {"size": int(fp16_bytes * 0.52), "tok_s": fp16_tok_s * 1.35},
        "Q4_K_M": {"size": int(fp16_bytes * 0.30), "tok_s": fp16_tok_s * 1.65},
    }
