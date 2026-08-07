import sys
sys.path.insert(0, ".")
from edge_mlx.fft import safe_fft_r2c, MockTensor
from edge_mlx.ops import native_unsupported_op
from edge_mlx.profiler import detect_silent_fallback


class DummyEvent:
    def __init__(self, name, device):
        self.name = name
        self.device = device


def test_safe_fft_r2c_returns_mps():
    t = MockTensor([1.0, 2.0, 3.0, 4.0], device="mps")
    res = safe_fft_r2c(t)
    assert res.device == "mps"


def test_native_unsupported_op_result():
    t = MockTensor([1.0, 2.0, 3.0], device="mps")
    res = native_unsupported_op(t)
    assert res.device == "mps"
    assert len(res.data) == 3


def test_detect_silent_fallback_catches_cpu():
    events = [DummyEvent("aten::add", "mps"), DummyEvent("aten::_fft_r2c", "cpu")]
    fb = detect_silent_fallback(events)
    assert len(fb) == 1
    assert fb[0] == "aten::_fft_r2c"
