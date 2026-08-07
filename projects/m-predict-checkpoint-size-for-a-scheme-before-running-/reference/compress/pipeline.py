import os
import json


def validate_compressed_checkpoint(model_dir, scheme):
    config_path = os.path.join(model_dir, "config.json")
    weights_path = os.path.join(model_dir, "model.safetensors")
    if not os.path.exists(config_path) or not os.path.exists(weights_path):
        return False
    with open(config_path, "r") as f:
        cfg = json.load(f)
    quant_config = cfg.get("quantization_config", {})
    if quant_config.get("format") != scheme.get("name"):
        return False
    return True
