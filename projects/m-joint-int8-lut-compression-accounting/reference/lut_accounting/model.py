def parse_config(config):
    layers = []
    for layer in config.get("layers", []):
        layers.append({
            "index": layer["index"],
            "type": layer["type"],
            "elements": layer["elements"],
            "bits": layer["bits"],
            "codebook_size": layer.get("codebook_size", 0)
        })
    return layers
