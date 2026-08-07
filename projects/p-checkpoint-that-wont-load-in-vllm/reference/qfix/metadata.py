def map_metadata(parsed):
    mapped = {}
    for k, v in parsed.items():
        new_k = k.replace("layers.", "model.layers.")
        mapped[new_k] = v
    return mapped
