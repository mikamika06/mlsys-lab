"""Grade fp16 LayerNorm computed via two-pass vs Welford variance.

The reference is computed here from a real NumPy oracle (fp32 truth + an fp16
re-implementation of each variance path). Nothing is hardcoded: on every fixture
the grader recomputes the fp32 ground-truth LayerNorm and both fp16 references,
then compares the candidate against them.
"""
import numpy as np

from mlsys import scorers

f16 = np.float16

# Ill-conditioned fixtures: (offset, spread, n, seed). Large common offset with a
# small spread pushes the two-pass running sum into the fp16 saturation regime,
# where Welford's bounded running mean wins. Kept so offset*n stays finite in fp16.
_SPECS = [
    (150.0, 0.4, 256, 11),
    (100.0, 0.3, 384, 12),
    (200.0, 0.5, 256, 13),
    (120.0, 0.4, 320, 14),
    (90.0, 0.3, 448, 15),
    (200.0, 0.5, 200, 16),
]


def _make_fixtures():
    out = []
    for off, sp, n, seed in _SPECS:
        rng = np.random.default_rng(seed)
        x = off + sp * rng.standard_normal(n)
        gamma = 1.0 + 0.1 * rng.standard_normal(n)
        beta = 0.1 * rng.standard_normal(n)
        out.append((x, gamma, beta))
    return out


def _welford_mean_var(x16):
    n = 0
    mean = f16(0.0)
    M2 = f16(0.0)
    for xi in x16:
        n += 1
        delta = f16(xi - mean)
        mean = f16(mean + f16(delta / f16(n)))
        delta2 = f16(xi - mean)
        M2 = f16(M2 + f16(delta * delta2))
    return mean, f16(M2 / f16(n))


def _two_pass_mean_var(x16):
    n = len(x16)
    s = f16(0.0)
    for xi in x16:
        s = f16(s + xi)
    mean = f16(s / f16(n))
    s2 = f16(0.0)
    for xi in x16:
        d = f16(xi - mean)
        s2 = f16(s2 + f16(d * d))
    return mean, f16(s2 / f16(n))


def _normalize(x16, mean, var, g16, b16, eps):
    denom = f16(np.sqrt(f16(var + f16(eps))))
    inv = f16(f16(1.0) / denom)
    out = np.empty(len(x16), dtype=f16)
    for i in range(len(x16)):
        xhat = f16(f16(x16[i] - mean) * inv)
        out[i] = f16(f16(g16[i] * xhat) + b16[i])
    return out


def _oracle(x, gamma, beta, which, eps=1e-5):
    x16 = np.asarray(x, dtype=f16)
    g16 = np.asarray(gamma, dtype=f16)
    b16 = np.asarray(beta, dtype=f16)
    if which == "welford":
        mean, var = _welford_mean_var(x16)
    else:
        mean, var = _two_pass_mean_var(x16)
    return _normalize(x16, mean, var, g16, b16, eps)


def _fp32_truth(x, gamma, beta, eps=1e-5):
    x = np.asarray(x, dtype=np.float64)
    g = np.asarray(gamma, dtype=np.float64)
    b = np.asarray(beta, dtype=np.float64)
    mean = x.mean()
    var = x.var()
    return g * (x - mean) / np.sqrt(var + eps) + b


_FAIL = {
    "welford_match_err": float("inf"),
    "two_pass_match_err": float("inf"),
    "err_ratio": float("inf"),
}


def grade(sol, fx) -> dict:
    welford_match = 0.0
    two_pass_match = 0.0
    worst_ratio = 0.0
    for x, gamma, beta in _make_fixtures():
        truth = _fp32_truth(x, gamma, beta)
        w_ref = np.asarray(_oracle(x, gamma, beta, "welford"), dtype=np.float64)
        t_ref = np.asarray(_oracle(x, gamma, beta, "two_pass"), dtype=np.float64)
        try:
            w_sol = np.asarray(sol.layernorm_fp16_welford(x, gamma, beta), dtype=np.float64)
            t_sol = np.asarray(sol.layernorm_fp16_two_pass(x, gamma, beta), dtype=np.float64)
        except Exception:
            return dict(_FAIL)
        if w_sol.shape != truth.shape or t_sol.shape != truth.shape:
            return dict(_FAIL)
        if not (np.all(np.isfinite(w_sol)) and np.all(np.isfinite(t_sol))):
            return dict(_FAIL)

        welford_match = max(welford_match, scorers.max_abs_err(w_sol, w_ref))
        two_pass_match = max(two_pass_match, scorers.max_abs_err(t_sol, t_ref))

        e_w = scorers.max_abs_err(w_sol, truth)
        e_t = scorers.max_abs_err(t_sol, truth)
        worst_ratio = max(worst_ratio, e_w / (e_t + 1e-12))

    return {
        "welford_match_err": float(welford_match),
        "two_pass_match_err": float(two_pass_match),
        "err_ratio": float(worst_ratio),
    }
