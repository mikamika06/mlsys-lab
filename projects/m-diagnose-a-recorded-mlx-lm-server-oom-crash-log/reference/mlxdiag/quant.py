def extract_quant_config(config_dict):
    """Reconstructs quantization bits and group_size from config.json structure."""
    quant = config_dict.get("quantization", {})
    if not isinstance(quant, dict):
        quant = {}

    bits = quant.get("bits")
    if bits is None:
        bits = config_dict.get("bits", 16)

    group_size = quant.get("group_size")
    if group_size is None:
        group_size = config_dict.get("group_size", 64)

    return {
        "bits": int(bits),
        "group_size": int(group_size),
        "is_quantized": int(bits) < 16,
    }
