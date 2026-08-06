import numpy as np


def inspect_config(config: dict) -> dict:
    required = ["lora_alpha", "peft_type", "r", "target_modules"]
    missing = sorted([k for k in required if k not in config])
    corrupted = []
    if "peft_type" in config and config["peft_type"] != "LORA":
        corrupted.append("peft_type")
    if "r" in config and (not isinstance(config["r"], int) or config["r"] <= 0):
        corrupted.append("r")
    if "lora_alpha" in config and (not isinstance(config["lora_alpha"], int) or config["lora_alpha"] <= 0):
        corrupted.append("lora_alpha")
    if "target_modules" in config and not isinstance(config["target_modules"], list):
        corrupted.append("target_modules")
    return {
        "valid": not missing and not corrupted,
        "missing": missing,
        "corrupted": sorted(corrupted)
    }


CONFIGS = [
    {"peft_type": "LORA", "r": 8, "lora_alpha": 16, "target_modules": ["q"]},
    {"r": 8, "lora_alpha": 16, "target_modules": ["q"]},
    {"peft_type": "LORA", "r": -1, "lora_alpha": 16, "target_modules": ["q"]},
    {"peft_type": "PROMPT_TUNING", "r": 8, "lora_alpha": 16, "target_modules": ["q"]},
    {"peft_type": "LORA", "r": 8, "lora_alpha": 16, "target_modules": "q_proj"},
    {"peft_type": "LORA", "r": 8, "lora_alpha": 0, "target_modules": ["q"]}
]
