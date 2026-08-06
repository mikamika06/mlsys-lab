import numpy as np


class AdapterRouter:
    def __init__(self, base_weights: dict):
        raise NotImplementedError

    def load_adapter(self, name: str, config: dict, lora_a: dict, lora_b: dict):
        raise NotImplementedError

    def set_adapter(self, name: str):
        raise NotImplementedError

    def forward(self, module_name: str, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError
