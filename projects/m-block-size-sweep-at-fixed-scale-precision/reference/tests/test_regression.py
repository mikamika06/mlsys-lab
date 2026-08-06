import numpy as np
from fp4quant.sweep import sweep_block_size


def test_block_size_sweep_argmin():
    np.random.seed(42)
    x = np.random.randn(256)
    block_sizes = [16, 32, 64, 128]
    idx, err = sweep_block_size(x, block_sizes)

    x_flat = x.astype(np.float64).ravel()
    errs = []
    for b in block_sizes:
        blocks = x_flat.reshape(-1, b)
        max_vals = np.max(np.abs(blocks), axis=1)
        scales = np.maximum(max_vals, 1e-30)
        log2_val = np.log2(scales)
        floor_exp = np.floor(log2_val)
        frac = log2_val - floor_exp
        exp = np.where(frac > 0.5, floor_exp + 1.0, np.where(frac < 0.5, floor_exp, np.where(np.abs(floor_exp) % 2 == 1.0, floor_exp + 1.0, floor_exp)))
        e8m0 = np.clip(exp + 127.0, 0, 255).astype(np.uint8)
        sc = np.power(2.0, e8m0.astype(np.float64) - 127.0)
        sc_exp = np.repeat(sc, b)
        q = np.clip(np.round(x_flat / np.maximum(sc_exp, 1e-30)), -7, 7)
        recon = q * sc_exp
        errs.append(float(np.mean((x_flat - recon) ** 2)))

    expected_idx = int(np.argmin(errs))
    assert idx == expected_idx, f"Expected index {expected_idx}, got {idx}"
