import json
import os


def inspect_and_verify_safetensors(dir_path, supported_architectures):
    """Verify safetensors metadata and architecture compatibility."""
    config_path = os.path.join(dir_path, "config.json")
    if not os.path.exists(config_path):
        return {"supported": False, "reason": "missing_config"}
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    archs = config.get("architectures", [])
    if not archs:
        arch = config.get("model_type")
        archs = [arch] if arch else []
    matching = [a for a in archs if a in supported_architectures]
    if not matching:
        return {"supported": False, "reason": "unsupported_architecture", "architectures": archs}
    has_weights = any(f.endswith(".safetensors") for f in os.listdir(dir_path))
    if not has_weights:
        return {"supported": False, "reason": "missing_safetensors_files"}
    return {
        "supported": True,
        "architecture": matching[0],
        "hidden_size": config.get("hidden_size", 0),
        "num_layers": config.get("num_hidden_layers", config.get("n_layer", 0)),
    }


def convert_safetensors_to_gguf_manifest(dir_path):
    """Build GGUF manifest dictionary from safetensors directory metadata."""
    config_path = os.path.join(dir_path, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return {
        "general.architecture": config.get("model_type", "llama"),
        "general.name": config.get("_name_or_path", "unknown"),
        "context_length": config.get("max_position_embeddings", 2048),
        "embedding_length": config.get("hidden_size", 4096),
        "block_count": config.get("num_hidden_layers", 32),
        "feed_forward_length": config.get("intermediate_size", 11008),
        "attention.head_count": config.get("num_attention_heads", 32),
    }
