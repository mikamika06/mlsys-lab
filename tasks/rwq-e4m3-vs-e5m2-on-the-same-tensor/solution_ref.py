import numpy as np


def _fp8_finite_grid(exp_bits: int, man_bits: int, reserve_top_exp_all: bool) -> np.ndarray:
    """All finite representable values of a generic OCP-style FP8 format."""
    bias = 2 ** (exp_bits - 1) - 1
    n_exp = 2 ** exp_bits
    n_man = 2 ** man_bits
    vals = set()
    for e in range(n_exp):
        top = (e == n_exp - 1)
        for m in range(n_man):
            if top:
                if reserve_top_exp_all:
                    continue
                if m == n_man - 1:
                    continue  # single reserved NaN code (E4M3FN style)
            if e == 0:
                v = (m / n_man) * 2.0 ** (1 - bias)
            else:
                v = (1.0 + m / n_man) * 2.0 ** (e - bias)
            vals.add(v)
            vals.add(-v)
    return np.array(sorted(vals), dtype=np.float64)


# E4M3: 4 exponent bits, 3 mantissa bits, only the top (e,m)=(max,max) code is NaN.
_E4M3_GRID = _fp8_finite_grid(4, 3, reserve_top_exp_all=False)
# E5M2: 5 exponent bits, 2 mantissa bits, the whole top exponent block is Inf/NaN.
_E5M2_GRID = _fp8_finite_grid(5, 2, reserve_top_exp_all=True)


def _cast_to_grid(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Snap x to the nearest value in `grid`; out-of-range values saturate
    to the nearest endpoint instead of producing Inf/NaN."""
    x = np.asarray(x, dtype=np.float64)
    flat = x.ravel()
    idx = np.searchsorted(grid, flat)
    idx = np.clip(idx, 1, len(grid) - 1)
    lo = grid[idx - 1]
    hi = grid[idx]
    choose_hi = (hi - flat) < (flat - lo)
    out = np.where(choose_hi, hi, lo)
    return out.reshape(x.shape)


def compare_fp8_formats(x: np.ndarray) -> tuple[float, float, str]:
    """Cast `x` to the E4M3 and E5M2 FP8 value grids, report each format's
    reconstruction MSE, and which one wins (smaller MSE; ties -> "e4m3")."""
    x64 = np.asarray(x, dtype=np.float64)
    x4 = _cast_to_grid(x64, _E4M3_GRID)
    x5 = _cast_to_grid(x64, _E5M2_GRID)
    mse_e4m3 = float(np.mean((x4 - x64) ** 2))
    mse_e5m2 = float(np.mean((x5 - x64) ** 2))
    winner = "e4m3" if mse_e4m3 <= mse_e5m2 else "e5m2"
    return mse_e4m3, mse_e5m2, winner
