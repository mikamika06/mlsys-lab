import numpy as np

from mlsys import scorers


def _create_dynamic_map(signed=True, max_exponent_bits=7, total_bits=8):
    data = []
    non_sign_bits = total_bits - (1 if signed else 0)
    for i in range(max_exponent_bits):
        fraction_items = int(2 ** (i + non_sign_bits - max_exponent_bits) + 1)
        boundaries = np.linspace(0.1, 1, fraction_items)
        means = (boundaries[:-1] + boundaries[1:]) / 2.0
        exp_val = 10.0 ** (-(max_exponent_bits - 1) + i)
        data.extend((exp_val * means).tolist())
        if signed:
            data.extend((-exp_val * means).tolist())
    data.append(0.0)
    data.append(1.0)
    data = sorted(data)
    arr = np.array(data, dtype=np.float64)
    assert arr.shape[0] == 2 ** total_bits
    return arr


_MAP = _create_dynamic_map()


def _blockwise_quant(v, blocksize, dynamic_map):
    v = np.asarray(v, dtype=np.float64)
    n = v.shape[0]
    n_blocks = int(np.ceil(n / blocksize)) if n > 0 else 0
    codes = np.zeros(n, dtype=np.uint8)
    absmax = np.zeros(n_blocks, dtype=np.float32)
    for b in range(n_blocks):
        lo, hi = b * blocksize, min((b + 1) * blocksize, n)
        block = v[lo:hi]
        amax = float(np.max(np.abs(block))) if block.size > 0 else 0.0
        amax = amax if amax > 0 else 1.0
        absmax[b] = amax
        normed = block / amax
        diffs = np.abs(normed[:, None] - dynamic_map[None, :])
        idx = np.argmin(diffs, axis=1)
        codes[lo:hi] = idx.astype(np.uint8)
    return codes, absmax


def _dequant(codes, absmax, blocksize, dynamic_map, n):
    v_hat = np.zeros(n, dtype=np.float64)
    n_blocks = absmax.shape[0]
    for b in range(n_blocks):
        lo, hi = b * blocksize, min((b + 1) * blocksize, n)
        v_hat[lo:hi] = dynamic_map[codes[lo:hi]] * absmax[b]
    return v_hat


def grade(sol, fx) -> dict:
    """
    Rebuilds the 256-entry bitsandbytes-style dynamic exponent map and does
    blockwise absmax quantize/dequantize with a NumPy oracle, on several
    random non-negative arrays and block sizes; compares the submission's
    dequantized reconstruction to the oracle's via relative L2 error.
    """
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(6):
        n = int(rng.integers(20, 200))
        blocksize = int(rng.choice([16, 32, 64]))
        v = (rng.random(n) ** 2) * rng.uniform(0.001, 100.0)
        try:
            got = sol.quantize_dequantize_v_blockwise(v.copy(), blocksize)
            v_hat_got, codes_got, absmax_got = got

            codes_exp, absmax_exp = _blockwise_quant(v, blocksize, _MAP)
            v_hat_exp = _dequant(codes_exp, absmax_exp, blocksize, _MAP, n)

            rel = scorers.rel_err(v_hat_exp, np.asarray(v_hat_got))
        except Exception:
            rel = float("inf")
        worst = max(worst, rel)
    return {"rel_err": worst}
