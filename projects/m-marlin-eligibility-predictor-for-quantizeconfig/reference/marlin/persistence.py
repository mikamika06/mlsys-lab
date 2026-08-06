import json

def save_quantize_config(config, path):
    payload = {
        "bits": config.get("bits"),
        "group_size": config.get("group_size", -1),
        "sym": config.get("sym", True),
        "desc_act": config.get("desc_act", False),
        "quant_method": config.get("quant_method", "marlin")
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def load_quantize_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
