import sys
sys.path.insert(0, ".")
import numpy as np
from int4.quant import quantize_weights

def test_quantization_roundtrip():
    w = np.random.randn(64, 64).astype(np.float32)
    packed, scale, shape = quantize_weights(w, group_size=64)
    even = packed & 0x0F
    odd = (packed >> 4) & 0x0F
    unpacked = np.empty(packed.size * 2, dtype=np.uint8)
    unpacked[0::2] = even
    unpacked[1::2] = odd
    unpacked = unpacked[:np.prod(shape)]
    q = unpacked.astype(np.int8) - 8
    dequant = (q.reshape(-1, 64).astype(np.float32) * scale).reshape(shape)
    assert np.mean(np.abs(w - dequant)) < 0.5

def test_scale_shape():
    w = np.random.randn(128, 128).astype(np.float32)
    _, scale, _ = quantize_weights(w, group_size=64)
    assert scale.shape[0] == 128 * 128 // 64
