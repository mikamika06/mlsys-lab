import numpy as np


class MockTensor:
    def __init__(self, data, device="mps"):
        self.data = np.array(data, dtype=np.float32)
        self.device = device


def native_unsupported_op(tensor):
    out_data = tensor.data * 2.0
    return MockTensor(out_data, device="mps")
