import numpy as np


def _swiglu_oracle(x, gate_w, up_w, down_w):
    gate = x.astype(np.float64) @ gate_w.astype(np.float64)
    up = x.astype(np.float64) @ up_w.astype(np.float64)
    silu = gate / (1.0 + np.exp(-gate))
    hidden = silu * up
    return hidden @ down_w.astype(np.float64)


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    cases = [
        (
            rng.normal(size=(3, 4)),
            rng.normal(size=(4, 6)),
            rng.normal(size=(4, 6)),
            rng.normal(size=(6, 4)),
        ),
        (
            rng.normal(size=(2, 3)),
            rng.normal(size=(3, 5)),
            rng.normal(size=(3, 5)),
            rng.normal(size=(5, 3)),
        ),
        (
            np.array([[0.0, 1.0, -1.0, 2.0]], dtype=np.float64),
            rng.normal(size=(4, 4)),
            rng.normal(size=(4, 4)),
            rng.normal(size=(4, 4)),
        ),
    ]

    worst = 0.0
    for x, gate_w, up_w, down_w in cases:
        ref = _swiglu_oracle(x, gate_w, up_w, down_w)
        try:
            got = sol.swiglu_ffn(
                x.tolist(),
                gate_w.tolist(),
                up_w.tolist(),
                down_w.tolist(),
            )
            got = np.asarray(got, dtype=np.float64)
            err = float(np.max(np.abs(ref - got)))
        except Exception:
            err = float("inf")
        worst = max(worst, err)

    return {"max_abs_err": worst}
