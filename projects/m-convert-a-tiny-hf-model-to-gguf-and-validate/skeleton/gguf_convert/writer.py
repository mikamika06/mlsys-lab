import numpy as np


class SimpleGGUFWriter:
    def __init__(self):
        raise NotImplementedError

    def add_architecture(self, name: str):
        raise NotImplementedError

    def add_kv(self, key: str, value):
        raise NotImplementedError

    def add_tensor(self, name: str, tensor: np.ndarray):
        raise NotImplementedError

    def write_to_dict(self) -> dict:
        raise NotImplementedError
