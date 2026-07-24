import numpy as np

from mlsys import scorers


def _ref_fold(W, b, gamma, beta, running_mean, running_var, eps):
    scale = gamma / np.sqrt(running_var + eps)
    W_folded = W * scale[:, None]
    b_folded = scale * (b - running_mean) + beta
    return W_folded, b_folded


def _bn(y, gamma, beta, mean, var, eps):
    return gamma * (y - mean) / np.sqrt(var + eps) + beta


def _scenarios():
    rng = np.random.default_rng(0)
    scenarios = []
    for out_f, in_f in [(4, 6), (8, 3), (2, 2), (10, 5)]:
        W = rng.normal(size=(out_f, in_f))
        b = rng.normal(size=out_f)
        gamma = rng.normal(size=out_f)
        gamma[0] = 0.01  # near-zero gamma channel
        beta = rng.normal(size=out_f)
        running_mean = rng.normal(size=out_f) * 3
        running_var = np.abs(rng.normal(size=out_f)) + 0.1
        eps = 1e-5
        xs = rng.normal(size=(5, in_f))
        scenarios.append((W, b, gamma, beta, running_mean, running_var, eps, xs))
    return scenarios


def grade(sol, fx) -> dict:
    worst = 0.0
    for W, b, gamma, beta, running_mean, running_var, eps, xs in _scenarios():
        Wf_ref, bf_ref = _ref_fold(W, b, gamma, beta, running_mean, running_var, eps)

        try:
            Wf_got, bf_got = sol.fold_bn_into_linear(
                W.copy(), b.copy(), gamma.copy(), beta.copy(),
                running_mean.copy(), running_var.copy(), eps,
            )
        except Exception:
            return {"max_abs_err": float("inf")}

        try:
            Wf_got = np.asarray(Wf_got, dtype=np.float64)
            bf_got = np.asarray(bf_got, dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}

        if Wf_got.shape != Wf_ref.shape or bf_got.shape != bf_ref.shape:
            return {"max_abs_err": float("inf")}

        err = max(
            scorers.max_abs_err(Wf_ref, Wf_got),
            scorers.max_abs_err(bf_ref, bf_got),
        )

        for x in xs:
            y = W @ x + b
            out_ref = _bn(y, gamma, beta, running_mean, running_var, eps)
            out_got = Wf_got @ x + bf_got
            err = max(err, scorers.max_abs_err(out_ref, out_got))

        if not np.isfinite(err):
            return {"max_abs_err": float("inf")}
        worst = max(worst, err)

    return {"max_abs_err": worst}
