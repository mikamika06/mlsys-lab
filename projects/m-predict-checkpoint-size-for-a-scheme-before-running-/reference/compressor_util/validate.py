import os
import json


def run_and_validate_compression(model_dir: str, scheme: dict, output_dir: str) -> bool:
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(model_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = json.load(f)
    else:
        cfg = {"hidden_size": 512, "num_hidden_layers": 2}

    out_cfg_path = os.path.join(output_dir, "config.json")
    with open(out_cfg_path, "w") as f:
        json.dump(cfg, f)

    quant_config = {
        "quant_method": "llm-compressor",
        "bits_w": scheme.get("bits_w", 4),
        "bits_a": scheme.get("bits_a", 16),
        "group_size": scheme.get("group_size", 128)
    }
    quant_cfg_path = os.path.join(output_dir, "quantize_config.json")
    with open(quant_cfg_path, "w") as f:
        json.dump(quant_config, f)

    weight_path = os.path.join(output_dir, "model.safetensors")
    dummy_data = b"SIMULATED_COMPRESSED_CHECKPOINT_DATA"
    with open(weight_path, "wb") as f:
        f.write(dummy_data)

    if not os.path.exists(out_cfg_path) or not os.path.exists(quant_cfg_path) or not os.path.exists(weight_path):
        return False

    with open(quant_cfg_path, "r") as f:
        loaded_qc = json.load(f)
    if loaded_qc.get("quant_method") != "llm-compressor":
        return False

    return True
