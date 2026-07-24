import numpy as np


def _ref_affine_quant_dequant(x: np.ndarray, bits: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n_levels = (1 << bits) - 1  # 2^bits - 1  (correct denominator)
    x_min = x.min()
    x_max = x.max()
    scale = (x_max - x_min) / n_levels if x_max != x_min else 1.0
    q = np.clip(np.round((x - x_min) / scale), 0, n_levels)
    return q * scale + x_min


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    cases = [
        rng.standard_normal(256),
        np.linspace(-5.0, 5.0, 512),
        np.array([0.0, 1.0, 2.0, 3.0]),
        rng.uniform(-100, 100, 1024),
    ]
    bits_list = [8, 4, 2, 8]

    worst_err = 0.0
    for x, bits in zip(cases, bits_list):
        x = x.astype(np.float64)
        ref = _ref_affine_quant_dequant(x, bits)
        try:
            got = np.asarray(sol.affine_quant_dequant(x.copy(), bits), dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}
        if got.shape != ref.shape:
            return {"max_abs_err": float("inf")}
        err = float(np.max(np.abs(got - ref)))
        if err > worst_err:
            worst_err = err

    return {"max_abs_err": worst_err}
