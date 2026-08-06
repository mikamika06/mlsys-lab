import numpy as np
from quantpack import pack_weights, unpack_weights, convert_awq_to_gptq
from quantpack.layout import get_packed_shape, get_memory_strides

def test_pack_unpack_roundtrip():
    for bits in [2, 3, 4, 8]:
        orig = np.random.randint(0, 1 << bits, size=64, dtype=np.int32)
        packed = pack_weights(orig, bits)
        unpacked = unpack_weights(packed, bits, len(orig))
        np.testing.assert_array_equal(orig, unpacked)

def test_layout_strides():
    shape = get_packed_shape(128, 128, 4)
    strides = get_memory_strides(shape, 4)
    assert len(strides) == len(shape)
    assert strides[-1] == 4
