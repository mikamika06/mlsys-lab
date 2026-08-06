import numpy as np
from qblocks.q4_0 import dequantize_q4_0, quantize_q4_0
from qblocks.q4_1 import dequantize_q4_1, quantize_q4_1
from qblocks.q5_1 import dequantize_q5_1, quantize_q5_1


def test_q4_0_roundtrip():
    data = np.linspace(-1.0, 1.0, 64, dtype=np.float32)
    blocks = quantize_q4_0(data)
    recovered = dequantize_q4_0(blocks)
    assert np.allclose(data, recovered, atol=0.2)


def test_q4_1_roundtrip():
    data = np.linspace(2.0, 10.0, 64, dtype=np.float32)
    blocks = quantize_q4_1(data)
    recovered = dequantize_q4_1(blocks)
    assert np.allclose(data, recovered, atol=0.6)


def test_q5_1_roundtrip():
    data = np.linspace(-5.0, 15.0, 64, dtype=np.float32)
    blocks = quantize_q5_1(data)
    recovered = dequantize_q5_1(blocks)
    assert np.allclose(data, recovered, atol=0.5)
