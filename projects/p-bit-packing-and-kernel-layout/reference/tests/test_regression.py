import numpy as np
from quant.packing import pack_bits, unpack_bits, simulate_kernel
from quant.layout import describe_layout, transform_layout

def test_layout_contract():
    info = describe_layout()
    assert isinstance(info, dict)
    assert "bits" in info

def test_roundtrip_basic():
    original = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32)
    packed = pack_bits(original, bits=4)
    unpacked = unpack_bits(packed, bits=4, shape=original.shape)
    transformed_orig = transform_layout(original)
    np.testing.assert_array_equal(unpacked.flatten()[:len(transformed_orig)], transformed_orig)

def test_kernel_simulation():
    original = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32)
    packed = pack_bits(original, bits=4)
    out = simulate_kernel(packed, scale=0.5)
    assert out.dtype == np.float32
    assert len(out) > 0
