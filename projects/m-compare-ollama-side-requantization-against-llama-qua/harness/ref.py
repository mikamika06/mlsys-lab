import json
import os
import numpy as np

SUPPORTED_ARCHS = ["LlamaForCausalLM", "MistralForCausalLM", "GemmaForCausalLM"]


def setup_test_directories(base_dir):
    d1 = os.path.join(base_dir, "model_valid")
    os.makedirs(d1, exist_ok=True)
    with open(os.path.join(d1, "config.json"), "w") as f:
        json.dump(
            {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "hidden_size": 2048,
                "num_hidden_layers": 16,
                "num_attention_heads": 16,
                "intermediate_size": 5632,
                "_name_or_path": "test-llama",
            },
            f,
        )
    with open(os.path.join(d1, "model.safetensors"), "wb") as f:
        f.write(b"MOCK_SAFETENSORS_DATA")

    d2 = os.path.join(base_dir, "model_unsupported")
    os.makedirs(d2, exist_ok=True)
    with open(os.path.join(d2, "config.json"), "w") as f:
        json.dump({"architectures": ["CustomUnknownArch"], "model_type": "custom"}, f)
    with open(os.path.join(d2, "weights.safetensors"), "wb") as f:
        f.write(b"MOCK_DATA")

    d3 = os.path.join(base_dir, "model_no_weights")
    os.makedirs(d3, exist_ok=True)
    with open(os.path.join(d3, "config.json"), "w") as f:
        json.dump({"architectures": ["MistralForCausalLM"]}, f)

    d4 = os.path.join(base_dir, "model_missing_config")
    os.makedirs(d4, exist_ok=True)

    return {"valid": d1, "unsupported": d2, "no_weights": d3, "missing_config": d4}
