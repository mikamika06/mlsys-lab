import numpy as np


class SimpleGGUFWriter:
    def __init__(self):
        self.kv = {}
        self.tensors = {}

    def add_architecture(self, name: str):
        self.kv["general.architecture"] = name

    def add_kv(self, key: str, value):
        self.kv[key] = value

    def add_tensor(self, name: str, tensor: np.ndarray):
        self.tensors[name] = np.ascontiguousarray(tensor)

    def write_to_dict(self) -> dict:
        return {
            "metadata": dict(self.kv),
            "tensors": dict(self.tensors)
        }
