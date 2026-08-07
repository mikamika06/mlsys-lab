HEADER_OVERHEAD = 320
PER_TENSOR_OVERHEAD = 64
ALIGNMENT_BYTES = 16


def _align(size, alignment=ALIGNMENT_BYTES):
    return (size + alignment - 1) & ~(alignment - 1)


def predict_quantized_size(model_spec, quant_mode="int8"):
    bits = 16 if quant_mode == "int16x8" else 8
    weight_bytes = sum((p["count"] * bits) // 8 for p in model_spec["weights"])
    return weight_bytes


def actual_flatbuffer_size(model_spec, quant_mode="int8"):
    bits = 16 if quant_mode == "int16x8" else 8
    total = HEADER_OVERHEAD
    for p in model_spec["weights"]:
        raw = (p["count"] * bits) // 8
        aligned = _align(raw)
        meta = PER_TENSOR_OVERHEAD
        if quant_mode in ("int8", "int16x8"):
            num_channels = p.get("channels", 1)
            meta += num_channels * (8 if quant_mode == "int16x8" else 4)
        total += aligned + meta
    return total
