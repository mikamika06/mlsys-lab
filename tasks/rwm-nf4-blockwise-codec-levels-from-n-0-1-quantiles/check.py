import numpy as np
from scipy.stats import norm


def _oracle_nf4_levels() -> np.ndarray:
    offset = 0.9677083
    v_pos = norm.ppf(np.linspace(offset, 0.5, 9)[:-1])       # 8 non-negative
    v_neg = -norm.ppf(np.linspace(offset, 0.5, 8)[:-1])       # 7 negative
    v = np.concatenate([v_pos, np.array([0.0]), v_neg])
    v = np.sort(v)
    v = v / np.max(np.abs(v))
    return v.astype(np.float64)


_LEVELS = _oracle_nf4_levels()


def _oracle_quantize(x: np.ndarray, block_size: int, levels: np.ndarray):
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    n_blocks = int(np.ceil(n / block_size))
    codes = np.zeros(n, dtype=np.uint8)
    absmax = np.zeros(n_blocks, dtype=np.float32)
    for b in range(n_blocks):
        s, e = b * block_size, min((b + 1) * block_size, n)
        block = x[s:e]
        am = float(np.max(np.abs(block)))
        if am == 0.0:
            am = 1.0
        absmax[b] = am
        norm_block = block / am
        d = np.abs(norm_block[:, None] - levels[None, :])
        codes[s:e] = np.argmin(d, axis=1).astype(np.uint8)
    return codes, absmax


def _oracle_dequantize(codes: np.ndarray, absmax: np.ndarray, block_size: int, levels: np.ndarray):
    n = codes.shape[0]
    out = np.zeros(n, dtype=np.float64)
    n_blocks = int(np.ceil(n / block_size))
    for b in range(n_blocks):
        s, e = b * block_size, min((b + 1) * block_size, n)
        out[s:e] = levels[codes[s:e]] * absmax[b]
    return out


def grade(sol, fx) -> dict:
    # --- codebook check ---
    try:
        got_levels = np.asarray(sol.nf4_levels(), dtype=np.float64).ravel()
        levels_err = float(np.max(np.abs(got_levels - _LEVELS))) if got_levels.shape == _LEVELS.shape else float("inf")
    except Exception:
        levels_err = float("inf")

    # --- round-trip check ---
    rng = np.random.default_rng(0)
    block_size = 64
    rel_errs = []
    for t in range(6):
        n = int(rng.integers(1000, 3000))
        x = rng.standard_normal(n).astype(np.float64)

        n_blocks = int(np.ceil(n / block_size))
        expected_packed_shape = ((n + 1) // 2,)
        expected_absmax_shape = (n_blocks,)

        try:
            packed, absmax = sol.quantize_4bit(x.copy(), block_size)
            packed = np.asarray(packed)
            absmax = np.asarray(absmax)

            if packed.dtype != np.uint8 or packed.shape != expected_packed_shape:
                rel_errs.append(float("inf"))
                continue
            if absmax.shape != expected_absmax_shape:
                rel_errs.append(float("inf"))
                continue

            x_hat = sol.dequantize_4bit(packed, absmax, n, block_size)
            x_hat = np.asarray(x_hat, dtype=np.float64).ravel()
            if x_hat.shape != (n,):
                rel_errs.append(float("inf"))
                continue
        except Exception:
            rel_errs.append(float("inf"))
            continue

        num = float(np.linalg.norm(x_hat - x))
        den = float(np.linalg.norm(x)) + 1e-12
        rel_errs.append(num / den)

    mean_rel_err = float(np.mean(rel_errs))

    return {
        "levels_max_abs_err": levels_err,
        "rel_err": mean_rel_err,
    }
