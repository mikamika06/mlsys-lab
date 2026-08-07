import numpy as np


class MockTensor:
    def __init__(self, data, device="mps"):
        self.data = np.array(data, dtype=np.float32)
        self.device = device


def safe_fft_r2c(tensor):
    try:
        if getattr(tensor, "device", "cpu") == "mps":
            pass
        return MockTensor(np.fft.rfftn(tensor.data), device="mps")
    except NotImplementedError:
        cpu_data = np.array(tensor.data, dtype=np.float32)
        res = np.fft.rfftn(cpu_data)
        return MockTensor(res, device="mps")
