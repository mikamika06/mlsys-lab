"""Oracle: independent re-implementation of QLoRA-style double quantization
of a first-level (block-64) NF4 absmax array, plus the bits/param it saves
versus storing every absmax as fp32.
"""
import numpy as np

from mlsys import scorers

FIRST_LEVEL_BLOCK_SIZE = 64


def _oracle(absmax, block_size):
    absmax = np.asarray(absmax, dtype=np.float64)
    n = absmax.shape[0]
    mean = float(np.mean(absmax))
    centered = absmax - mean

    n_blocks2 = -(-n // block_size)
    recon = np.zeros(n, dtype=np.float64)
    for b in range(n_blocks2):
        lo = b * block_size
        hi = min(lo + block_size, n)
        seg = centered[lo:hi]
        amax = float(np.max(np.abs(seg)))
        scale = amax / 127.0 if amax > 0 else 1.0
        c = np.clip(np.round(seg / scale), -127, 127)
        recon[lo:hi] = c * scale + mean

    original_bits = 32.0 * n
    new_bits = 8.0 * n + 32.0 * n_blocks2 + 32.0
    total_params = float(n) * FIRST_LEVEL_BLOCK_SIZE
    bits_saved_per_param = (original_bits - new_bits) / total_params

    return recon, bits_saved_per_param


def grade(sol, fx) -> dict:
    cases = [
        (np.asarray(fx["nf4_absmax"], dtype=np.float64), 256),
    ]
    rng = np.random.default_rng(3)
    extra = np.abs(rng.standard_normal(4096)).astype(np.float64) * 0.05
    cases.append((extra, 128))

    worst_rel = 0.0
    worst_bits = 0.0
    for absmax, block_size in cases:
        recon_ref, bits_ref = _oracle(absmax, block_size)
        try:
            out = sol.double_quantize_absmax(absmax.copy(), block_size)
            _codes, _scales, _mean, recon_got, bits_got = out
            recon_got = np.asarray(recon_got, dtype=np.float64)
        except Exception:
            return {"rel_err": float("inf"), "bits_abs_err": float("inf")}

        if recon_got.shape != recon_ref.shape or not np.all(np.isfinite(recon_got)):
            return {"rel_err": float("inf"), "bits_abs_err": float("inf")}

        worst_rel = max(worst_rel, scorers.rel_err(recon_ref, recon_got))
        worst_bits = max(worst_bits, abs(float(bits_got) - bits_ref))

    return {"rel_err": worst_rel, "bits_abs_err": worst_bits}
