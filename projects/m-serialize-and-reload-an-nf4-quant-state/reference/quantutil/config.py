import json

def rebuild_quantization_config(safetensors_metadata):
    config = {}
    for k, v in safetensors_metadata.items():
        if k.startswith("quant_method"):
            config["quant_method"] = v
        if k.startswith("bits"):
            config["bits"] = int(v)
        if k.startswith("bnb_4bit_quant_type"):
            config["bnb_4bit_quant_type"] = v
        if k.startswith("bnb_4bit_compute_dtype"):
            config["bnb_4bit_compute_dtype"] = v
        if k.startswith("bnb_4bit_use_double_quant"):
            config["bnb_4bit_use_double_quant"] = v.lower() == "true"
    if not config:
        if "quantization_config" in safetensors_metadata:
            config = json.loads(safetensors_metadata["quantization_config"])
    return config
