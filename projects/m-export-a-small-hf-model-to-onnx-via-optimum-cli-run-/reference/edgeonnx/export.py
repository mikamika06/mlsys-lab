import numpy as np


def parse_export_config(config_dict):
    return {
        "name": config_dict.get("model_name", "unknown"),
        "hidden_size": config_dict.get("hidden_size", 64),
        "num_layers": config_dict.get("num_layers", 2),
        "vocab_size": config_dict.get("vocab_size", 256)
    }


def simulate_export(model_spec):
    nodes = []
    for i in range(model_spec.get("num_layers", 2)):
        nodes.append({"name": f"Layer{i}_Attn", "op": "Attention", "provider": "CoreMLExecutionProvider"})
        if i == 0:
            nodes.append({"name": f"Layer{i}_Custom", "op": "CustomOp", "provider": "CPUExecutionProvider"})
    return {
        "name": model_spec.get("name", "model"),
        "nodes": nodes,
        "weights": np.ones((model_spec.get("vocab_size", 256), model_spec.get("hidden_size", 64)), dtype=np.float32) * 0.1
    }
