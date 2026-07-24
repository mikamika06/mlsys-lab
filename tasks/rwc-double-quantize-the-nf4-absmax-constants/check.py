import numpy as np


def _nf4_codebook():
    """The 16 NF4 levels, reconstructed empirically as the quantiles of a
    standard normal sample (matches the theoretical asymmetric NF4 grid
    used by QLoRA)."""
    rng = np.random.RandomState(0)
    sample = rng.randn(1_000_000)
    q = (np.arange(16) + 0.5) / 16
    return np.quantile(sample, q)


_CODEBOOK = _nf4_codebook()


def _nf4_quantize(flat, block_size):
    """Level-1 NF4 blockwise quantization: one fp32 absmax per block."""
    n = len(flat)
    n_blocks = int(np.ceil(n / block_size))
    codes = np.empty(n, dtype=np.uint8)
    c1 = np.empty(n_blocks, dtype=np.float64)
    for bi, i in enumerate(range(0, n, block_size)):
        blk = flat[i:i + block_size]
        m = float(np.max(np.abs(blk)))
        if m == 0.0:
            m = 1.0
        c1[bi] = m
        y = blk / m
        idx = np.abs(y[:, None] - _CODEBOOK[None, :]).argmin(axis=1)
        codes[i:i + block_size] = idx.astype(np.uint8)
    return codes, c1


def _affine8_quant_dequant(x, outer_block):
    """Level-2: asymmetric (affine) min-max 8-bit blockwise quantization
    of the level-1 absmax constants themselves."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    out = np.empty(n, dtype=np.float64)
    for i in range(0, n, outer_block):
        grp = x[i:i + outer_block]
        lo, hi = float(np.min(grp)), float(np.max(grp))
        scale = (hi - lo) / 255.0
        if scale == 0.0:
            scale = 1.0
        zp = round(-lo / scale)
        code = np.clip(np.round(grp / scale + zp), 0, 255)
        out[i:i + outer_block] = (code - zp) * scale
    return out


def _oracle(weights, block_size, outer_block):
    flat = np.asarray(weights, dtype=np.float64).ravel()
    n = len(flat)

    codes, c1 = _nf4_quantize(flat, block_size)
    c1_hat = _affine8_quant_dequant(c1, outer_block)

    block_idx = np.arange(n) // block_size
    recon = (_CODEBOOK[codes] * c1_hat[block_idx]).reshape(np.asarray(weights).shape)

    bits_per_param = 4.0 + 8.0 / block_size + 32.0 / (block_size * outer_block)
    return recon, bits_per_param


def _cases():
    rng = np.random.default_rng(0)
    cases = []
    cases.append((rng.standard_normal(1000), 64, 4))
    cases.append((rng.standard_normal(5000), 64, 8))
    cases.append((rng.standard_normal((300, 7)), 32, 3))
    cases.append((np.zeros(256), 64, 2))  # degenerate all-zero block
    cases.append((rng.standard_normal(777), 128, 5))  # not evenly divisible
    return cases


def grade(sol, fx) -> dict:
    worst = 0.0
    for weights, block_size, outer_block in _cases():
        ref_recon, ref_bits = _oracle(weights, block_size, outer_block)
        try:
            got_recon, got_bits = sol.nf4_double_quant_dequant(
                weights.copy(), block_size, outer_block
            )
            got_recon = np.asarray(got_recon, dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}

        if got_recon.shape != ref_recon.shape:
            return {"max_abs_err": float("inf")}

        err_recon = float(np.max(np.abs(got_recon - ref_recon)))
        err_bits = abs(float(got_bits) - ref_bits)
        worst = max(worst, err_recon, err_bits)

    return {"max_abs_err": worst}
