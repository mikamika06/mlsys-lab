import numpy as np


def _fp8_finite_grid(exp_bits: int, man_bits: int, reserve_top_exp_all: bool) -> np.ndarray:
    """All finite representable values of a generic OCP-style FP8 format.

    exp_bits/man_bits: field widths. bias = 2**(exp_bits-1) - 1.
    reserve_top_exp_all: True -> the entire top exponent block is reserved
        for Inf/NaN (E5M2 style). False -> only the single top-exponent,
        all-ones-mantissa code is NaN, every other top-exponent code is a
        normal finite value ("FN" style, E4M3).
    """
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
                    continue  # single reserved NaN code
            if e == 0:
                v = (m / n_man) * 2.0 ** (1 - bias)
            else:
                v = (1.0 + m / n_man) * 2.0 ** (e - bias)
            vals.add(v)
            vals.add(-v)
    return np.array(sorted(vals), dtype=np.float64)


_E4M3_GRID = _fp8_finite_grid(4, 3, reserve_top_exp_all=False)
_E5M2_GRID = _fp8_finite_grid(5, 2, reserve_top_exp_all=True)


def _cast_to_grid(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Snap every element of x to the nearest value in the sorted `grid`.

    Values outside the grid's range saturate to the nearest endpoint
    (never Inf/NaN) — a storage-cast quantizer, not an IEEE overflow cast.
    """
    x = np.asarray(x, dtype=np.float64)
    flat = x.ravel()
    idx = np.searchsorted(grid, flat)
    idx = np.clip(idx, 1, len(grid) - 1)
    lo = grid[idx - 1]
    hi = grid[idx]
    choose_hi = (hi - flat) < (flat - lo)
    out = np.where(choose_hi, hi, lo)
    return out.reshape(x.shape)


def _oracle_compare(x: np.ndarray):
    x64 = np.asarray(x, dtype=np.float64)
    x4 = _cast_to_grid(x64, _E4M3_GRID)
    x5 = _cast_to_grid(x64, _E5M2_GRID)
    mse4 = float(np.mean((x4 - x64) ** 2))
    mse5 = float(np.mean((x5 - x64) ** 2))
    winner = "e4m3" if mse4 <= mse5 else "e5m2"
    return mse4, mse5, winner


def _make_cases(rng: np.random.Generator):
    cases = []

    # narrow, tightly-clustered values -> the finer E4M3 mantissa grid should win
    cases.append(rng.normal(0.0, 0.05, size=200))

    # mostly-narrow with a handful of huge outliers -> E4M3 saturates hard,
    # E5M2's bigger exponent range should win
    wide = rng.normal(0.0, 1.0, size=200)
    out_idx = rng.choice(200, size=5, replace=False)
    wide[out_idx] = rng.uniform(2000.0, 50000.0, size=5) * rng.choice([-1.0, 1.0], size=5)
    cases.append(wide)

    # moderate range, no outliers, well inside both formats' max magnitude
    cases.append(rng.uniform(-300.0, 300.0, size=150))

    # hand-picked edge values: zeros, near-max, far-out-of-range
    cases.append(np.array([
        0.0, -0.0, 1.0, -1.0, 3.5, 500.0, -500.0,
        100000.0, -100000.0, 0.001, -0.001, 448.0, 57344.0,
    ], dtype=np.float64))

    return cases


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    cases = _make_cases(rng)

    max_mse_err = 0.0
    winner_hits = 0
    total = len(cases)

    for x in cases:
        exp_mse4, exp_mse5, exp_winner = _oracle_compare(x)
        try:
            got = sol.compare_fp8_formats(np.array(x, dtype=np.float64, copy=True))
            got_mse4, got_mse5, got_winner = got
            got_mse4 = float(got_mse4)
            got_mse5 = float(got_mse5)
            got_winner = str(got_winner)
        except Exception:
            max_mse_err = float("inf")
            continue

        err = max(abs(got_mse4 - exp_mse4), abs(got_mse5 - exp_mse5))
        if not np.isfinite(err):
            err = float("inf")
        max_mse_err = max(max_mse_err, err)
        if got_winner == exp_winner:
            winner_hits += 1

    return {
        "mse": max_mse_err,
        "exact_match": winner_hits / total,
    }
