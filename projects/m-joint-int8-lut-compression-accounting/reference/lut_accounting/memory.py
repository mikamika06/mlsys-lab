def calculate_bytes(config, alignment):
    total = 0
    for layer in config.get("layers", []):
        elements = layer["elements"]
        bits = layer["bits"]
        if bits == 8 and layer["type"] == "int8":
            raw = elements
        elif layer["type"] == "lut":
            cb_size = layer.get("codebook_size", 256)
            cb_bytes = cb_size * 4
            idx_bits = bits * elements
            raw = cb_bytes + (idx_bits + 7) // 8
        else:
            raw = (elements * bits + 7) // 8
        padded = ((raw + alignment - 1) // alignment) * alignment
        total += padded
    return total


def uniform_baseline_bytes(config, alignment):
    total = 0
    for layer in config.get("layers", []):
        elements = layer["elements"]
        raw = elements
        padded = ((raw + alignment - 1) // alignment) * alignment
        total += padded
    return total
