def build_variants(weights):
    base_size = sum(w.nbytes for w in weights.values())
    return {
        "fp32": {"size": base_size, "io_dtype": "float32", "quantized": False},
        "fp16": {"size": base_size // 2, "io_dtype": "float16", "quantized": False},
        "dynamic": {"size": base_size // 4, "io_dtype": "float32", "quantized": True},
        "int8_full": {"size": base_size // 4, "io_dtype": "int8", "quantized": True}
    }
