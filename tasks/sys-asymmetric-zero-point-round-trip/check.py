import numpy as np


def _oracle(x, qmin, qmax):
    mn = min(0.0, float(np.min(x)))
    mx = max(0.0, float(np.max(x)))
    scale = (mx - mn) / (qmax - qmin) if mx > mn else 1.0
    zp = int(np.clip(round(qmin - mn / scale), qmin, qmax))
    codes = np.clip(np.round(x / scale + zp), qmin, qmax)
    deq = (codes - zp) * scale
    return deq, scale


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    worst_abs = 0.0
    worst_ratio = 0.0

    for _ in range(8):
        n = int(rng.integers(2, 60))
        x = rng.normal(scale=float(rng.uniform(0.1, 5.0)), size=n)
        qmin, qmax = 0, 255

        deq_exp, scale = _oracle(x, qmin, qmax)

        try:
            deq_got = np.asarray(sol.affine_quant_dequant(x.copy(), qmin, qmax), dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf"), "max_step_ratio": float("inf")}

        if deq_got.shape != deq_exp.shape:
            return {"max_abs_err": float("inf"), "max_step_ratio": float("inf")}

        err = float(np.max(np.abs(deq_got - deq_exp)))
        worst_abs = max(worst_abs, err)

        ratio = float(np.max(np.abs(x - deq_got)) / scale)
        worst_ratio = max(worst_ratio, ratio)

    return {"max_abs_err": worst_abs, "max_step_ratio": worst_ratio}
