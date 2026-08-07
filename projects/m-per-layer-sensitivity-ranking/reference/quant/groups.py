def emit_config_groups(model_config, assigned_bits):
    buckets = {}
    for layer, bits in zip(model_config["layers"], assigned_bits):
        buckets.setdefault(bits, []).append(layer["index"])

    groups = []
    for bits, layers in sorted(buckets.items()):
        groups.append({
            "bits": int(bits),
            "layers": sorted(layers)
        })
    return groups
