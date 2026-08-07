import sys
import numpy as np

sys.path.insert(0, ".")
from triton_mask.kernel import compute_block_mask, process_data, run_boundary_sweep


def test_tail_mask_bounds():
    offsets = np.arange(128, dtype=np.int32)
    N = 100
    mask = compute_block_mask(offsets, N)
    assert np.all(mask[:100])
    assert not np.any(mask[100:])


def test_non_multiple_block_size():
    N = 75
    BLOCK_SIZE = 32
    x = np.ones(N, dtype=np.float32) * 2.0
    out = process_data(x, N, BLOCK_SIZE)
    expected = x * 2.5 + 1.0
    assert out.shape == (N,)
    assert np.allclose(out, expected)


def test_boundary_sweep():
    results = run_boundary_sweep(1, 100, BLOCK_SIZE=16)
    for n, ok in results.items():
        assert ok
