import sys
import numpy as np

sys.path.insert(0, ".")
from sparsecoder.codec import encode_bitmask_values, decode_bitmask_values
from sparsecoder.analysis import breakeven_sparsity, measure_byte_savings


def test_round_trip_integrity():
    arr = np.array([1.5, 0.0, 0.0, -2.1, 0.0, 3.3, 0.0, 0.0, 4.4, 0.0], dtype=np.float32)
    encoded = encode_bitmask_values(arr, block_size=4)
    decoded = decode_bitmask_values(encoded, arr.shape, block_size=4)
    np.testing.assert_allclose(arr, decoded)


def test_breakeven_bounds():
    shape = (64, 64)
    be = breakeven_sparsity(shape, dtype_bytes=2)
    assert 0.0 <= be <= 1.0


def test_measure_byte_savings_keys():
    shape = (32, 32)
    sparsities = [0.5, 0.8]
    res = measure_byte_savings(shape, sparsities, dtype_bytes=2, block_size=8)
    for s in sparsities:
        assert s in res
        assert "savings_percent" in res[s]
