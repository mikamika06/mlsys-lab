import numpy as np
from quantpack.packing import pack_weights, unpack_weights
from quantpack.convert import convert_awq_to_gptq
from quantpack.shapes import packed_shape_and_stride

def test_roundtrip_2bit():
    mat = np.random.randint(0, 4, size=(16, 16), dtype=np.int32)
    packed = pack_weights(mat, 2)
    unpacked = unpack_weights(packed, 2, mat.shape)
    assert np.array_equal(mat, unpacked)

def test_roundtrip_3bit():
    mat = np.random.randint(0, 8, size=(10, 10), dtype=np.int32)
    packed = pack_weights(mat, 3)
    unpacked = unpack_weights(packed, 3, mat.shape)
    assert np.array_equal(mat, unpacked)

def test_roundtrip_4bit():
    mat = np.random.randint(0, 16, size=(8, 8), dtype=np.int32)
    packed = pack_weights(mat, 4)
    unpacked = unpack_weights(packed, 4, mat.shape)
    assert np.array_equal(mat, unpacked)

def test_roundtrip_8bit():
    mat = np.random.randint(0, 256, size=(4, 4), dtype=np.int32)
    packed = pack_weights(mat, 8)
    unpacked = unpack_weights(packed, 8, mat.shape)
    assert np.array_equal(mat, unpacked)
