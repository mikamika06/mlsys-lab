def infer_scheme_from_index(index_data):
    weight_map = index_data.get("weight_map", {})
    keys = list(weight_map.keys())
    has_packed = any(".weight_packed" in k or ".packed_weight" in k for k in keys)
    has_scale = any(".weight_scale" in k or ".scale" in k for k in keys)
    has_zero_point = any(".zero_point" in k or ".zp" in k for k in keys)

    metadata = index_data.get("metadata", {})
    quant_config = metadata.get("quantization_config", {})
    format_str = quant_config.get("format", "").lower()

    if "pack" in format_str or has_packed:
        return "pack-quantized"
    if "int" in format_str or (has_scale and not has_packed):
        return "int-quantized"
    return "unquantized"
