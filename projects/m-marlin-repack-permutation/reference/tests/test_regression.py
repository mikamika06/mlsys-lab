import numpy as np
from repack.pack import pack_int4_standard, repack_gptq_to_marlin
from repack.permute import permute_weights, unpermute_weights
from repack.unpack import marlin_to_gptq


def test_roundtrip_marlin_repack():
    rng = np.random.default_rng(123)
    for K, N in [(16, 64), (32, 128), (64, 256)]:
        W = rng.integers(0, 16, size=(K, N), dtype=np.uint8)
        std_packed = pack_int4_standard(W)
        marlin_packed = repack_gptq_to_marlin(std_packed, K, N)
        recovered_gptq = marlin_to_gptq(marlin_packed, K, N)
        np.testing.assert_array_equal(recovered_gptq, std_packed)


def test_permute_unpermute_identity():
    rng = np.random.default_rng(456)
    for K, N in [(16, 64), (32, 64)]:
        W = rng.integers(0, 16, size=(K, N), dtype=np.uint8)
        W_perm = permute_weights(W)
        W_restored = unpermute_weights(W_perm, K, N)
        np.testing.assert_array_equal(W_restored, W)


def test_value_histogram_preserved():
    rng = np.random.default_rng(789)
    W = rng.integers(0, 16, size=(32, 128), dtype=np.uint8)
    W_perm = permute_weights(W)
    orig_counts = np.bincount(W.ravel(), minlength=16)
    perm_counts = np.bincount(W_perm.ravel(), minlength=16)
    np.testing.assert_array_equal(orig_counts, perm_counts)
