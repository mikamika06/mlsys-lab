from quantizer.config import analyze_retention_reasons


def search_per_layer_config(config):
    reasons = analyze_retention_reasons(config)
    layers = sorted(config["layers"], key=lambda x: x["params"], reverse=True)
    current_bytes = sum(l["params"] * 2 for l in layers)
    selected = {}
    for l in layers:
        if reasons.get(l["name"]) == "quantized":
            potential_saving = l["params"] * 1
            if current_bytes - potential_saving <= config["budget_bytes"]:
                selected[l["name"]] = "int8"
                current_bytes -= potential_saving
            else:
                selected[l["name"]] = "fp16"
        else:
            selected[l["name"]] = "fp16"
    return selected
