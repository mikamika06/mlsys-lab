import numpy as np
from peft_mechanics.config import inspect_config


class AdapterRouter:
    def __init__(self, base_weights: dict):
        self.base = base_weights
        self.adapters = {}
        self.active = None

    def load_adapter(self, name: str, config: dict, lora_a: dict, lora_b: dict):
        info = inspect_config(config)
        if not info["valid"]:
            raise ValueError("invalid config")
        self.adapters[name] = {"config": config, "a": lora_a, "b": lora_b}
        if self.active is None:
            self.active = name

    def set_adapter(self, name: str):
        if name not in self.adapters:
            raise ValueError("not found")
        self.active = name

    def forward(self, module_name: str, x: np.ndarray) -> np.ndarray:
        out = self.base[module_name] @ x
        if self.active is not None:
            adapter = self.adapters[self.active]
            if module_name in adapter["config"]["target_modules"]:
                a = adapter["a"][module_name]
                b = adapter["b"][module_name]
                scaling = adapter["config"]["lora_alpha"] / adapter["config"]["r"]
                out = out + (b @ (a @ x)) * scaling
        return out
