import numpy as np


class MultiAdapterLinear:
    def __init__(self, in_features, out_features, base_weight=None):
        self.in_features = in_features
        self.out_features = out_features
        if base_weight is not None:
            self.base_weight = np.array(base_weight, dtype=np.float32)
        else:
            self.base_weight = np.zeros((out_features, in_features), dtype=np.float32)
        self.adapters = {}
        self.active_adapter = None

    def add_adapter(self, adapter_name, r, lora_alpha, lora_A=None, lora_B=None):
        if lora_A is None:
            lora_A = np.zeros((r, self.in_features), dtype=np.float32)
        else:
            lora_A = np.array(lora_A, dtype=np.float32)
        if lora_B is None:
            lora_B = np.zeros((self.out_features, r), dtype=np.float32)
        else:
            lora_B = np.array(lora_B, dtype=np.float32)
        scaling = float(lora_alpha) / float(r)
        self.adapters[adapter_name] = {
            "r": r,
            "lora_alpha": lora_alpha,
            "scaling": scaling,
            "lora_A": lora_A,
            "lora_B": lora_B,
        }
        if self.active_adapter is None:
            self.active_adapter = adapter_name

    def set_adapter(self, adapter_name):
        if adapter_name not in self.adapters:
            raise KeyError(f"Adapter '{adapter_name}' not found in registered adapters: {sorted(self.adapters.keys())}")
        self.active_adapter = adapter_name

    def forward(self, x):
        x = np.array(x, dtype=np.float32)
        base_out = x @ self.base_weight.T
        if self.active_adapter is None or self.active_adapter not in self.adapters:
            return base_out
        ad = self.adapters[self.active_adapter]
        delta = (x @ ad["lora_A"].T) @ ad["lora_B"].T
        return base_out + delta * ad["scaling"]
