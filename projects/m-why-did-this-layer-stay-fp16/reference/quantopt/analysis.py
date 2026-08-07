def why_fp16(layer):
    if 16 in layer.get("supported_bits", []) and layer.get("sensitivity", 0.0) > 0.5:
        return "high_sensitivity"
    if len(layer.get("supported_bits", [])) == 1:
        return "unsupported_low_bit"
    return "none"
