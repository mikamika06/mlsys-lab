CONFIGS = [
    {"r": 8, "hidden_size": 768, "intermediate_size": 3072},
    {"r": 16, "hidden_size": 1024, "intermediate_size": 4096},
    {"r": 32, "hidden_size": 2048, "intermediate_size": 8192}
]

TARGET_MODULES = [
    ["q_proj", "v_proj"],
    ["q_proj", "k_proj", "v_proj", "o_proj"],
    ["gate_proj", "up_proj", "down_proj"]
]

def compute_adapter_bytes(config, target_modules, dtype_bytes=2):
    r = config.get("r", 8)
    total_elements = 0
    for mod_name in target_modules:
        if "q_proj" in mod_name or "k_proj" in mod_name or "v_proj" in mod_name or "o_proj" in mod_name:
            in_features = config.get("hidden_size", 768)
            out_features = config.get("hidden_size", 768)
        else:
            in_features = config.get("hidden_size", 768)
            out_features = config.get("intermediate_size", 3072)
        lora_A = r * in_features
        lora_B = out_features * r
        total_elements += lora_A + lora_B
    return total_elements * dtype_bytes

def compute_storage_ratio(adapter_bytes, base_model_bytes):
    if base_model_bytes == 0:
        return 0.0
    return float(adapter_bytes) / float(base_model_bytes)
