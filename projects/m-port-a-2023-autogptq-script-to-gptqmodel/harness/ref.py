CONFIGS = [
    {"bits": 4, "group_size": 128, "desc_act": False, "sym": True, "damp_percent": 0.1},
    {"bits": 8, "group_size": -1, "desc_act": True, "sym": False, "damp_percent": 0.05},
    {"bits": 3, "group_size": 32, "desc_act": False, "sym": True, "damp_percent": 0.2}
]

ORACLE_TESTS = [
    ({"bits": 4, "group_size": 128, "desc_act": False}, "exllamav2", True),
    ({"bits": 5, "group_size": 128, "desc_act": False}, "exllamav2", False),
    ({"bits": 4, "group_size": 64, "desc_act": False}, "exllamav2", False),
    ({"bits": 4, "group_size": 128, "desc_act": True}, "triton", True),
    ({"bits": 2, "group_size": 128, "desc_act": True}, "triton", False),
    ({"bits": 4, "group_size": 128, "desc_act": False}, "gptqmodel", True)
]

FILE_LISTINGS = [
    (["config.json", "quantize_config.json", "model.safetensors", "gptqmodel_metadata.json"], "gptqmodel"),
    (["config.json", "quantize_config.json", "pytorch_model.bin", "autogptq_internal.py"], "autogptq"),
    (["config.json", "model.safetensors"], "transformers")
]
