def export_int8_io(weights):
    base_size = sum(w.nbytes for w in weights.values())
    return {
        "size": base_size // 4,
        "io_dtype": "int8",
        "quantized": True
    }
