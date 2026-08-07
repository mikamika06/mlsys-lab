import numpy as np


class MockTensor:
    def __init__(self, data, device="mps"):
        self.data = np.array(data, dtype=np.float32)
        self.device = device


class MockProfilerEvent:
    def __init__(self, name, device):
        self.name = name
        self.device = device


def run_fft_case(x, use_fallback=True):
    if not use_fallback:
        raise NotImplementedError("aten::_fft_r2c is not implemented for MPS")
    return MockTensor(np.fft.rfftn(x.data), device="mps")


def run_native_op(x):
    return MockTensor(x.data * 2.0, device="mps")


def detect_fallbacks(events):
    fallbacks = []
    for ev in events:
        if ev.device != "mps":
            fallbacks.append(ev.name)
    return fallbacks
