def analyze_retention_reasons(config):
    reasons = {}
    for layer in config["layers"]:
        if layer.get("sensitive", False):
            reasons[layer["name"]] = "sensitive_exclusion"
        elif layer["type"] == "embedding":
            reasons[layer["name"]] = "unsupported_op_type"
        elif layer["params"] < 4096:
            reasons[layer["name"]] = "below_size_threshold"
        else:
            reasons[layer["name"]] = "quantized"
    return reasons
