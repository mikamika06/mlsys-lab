import numpy as np

from mlsys import scorers


def _e4m3_roundtrip(x, scale):
    y = np.asarray(x, dtype=np.float64) / scale
    y = np.clip(y, -448.0, 448.0)
    sign = np.sign(y)
    ay = np.abs(y)

    exp = np.floor(np.log2(np.maximum(ay, 2 ** -9)))
    exp = np.clip(exp, -6, 7)
    frac = ay / (2.0 ** exp) - 1.0
    mant = np.round(frac * 8.0) / 8.0
    val = (1.0 + mant) * (2.0 ** exp)
    val = np.where(ay < 2 ** -6, np.round(ay / (2 ** -9)) * (2 ** -9), val)
    val = np.where(ay == 0, 0.0, val)

    return sign * val * scale


def _oracle(K, V):
    sk = max(float(np.max(np.abs(K))) / 448.0, 1e-12)
    sv = max(float(np.max(np.abs(V))) / 448.0, 1e-12)
    K_hat = _e4m3_roundtrip(K, sk)
    V_hat = _e4m3_roundtrip(V, sv)
    mse_k = float(np.mean((K_hat - K) ** 2))
    mse_v = float(np.mean((V_hat - V) ** 2))
    return mse_k, mse_v


def _synthetic_cases():
    rng = np.random.default_rng(73)
    cases = []
    for _ in range(4):
        shape = (int(rng.integers(8, 40)), int(rng.integers(4, 20)))
        K = rng.standard_normal(shape) * rng.uniform(0.1, 5.0)
        V = rng.standard_normal(shape) * rng.uniform(0.1, 5.0)
        cases.append((K, V))
    return cases


def grade(sol, fx) -> dict:
    cases = [(fx["k"], fx["v"])] + _synthetic_cases()

    worst = 0.0
    for K, V in cases:
        ref_k, ref_v = _oracle(K, V)
        try:
            got = sol.kv_fp8_reconstruction_mse(K.tolist(), V.tolist())
            got_k = float(got["mse_k"])
            got_v = float(got["mse_v"])
        except Exception:
            return {"rel_err": float("inf")}

        err = scorers.rel_err(np.array([ref_k, ref_v]), np.array([got_k, got_v]))
        worst = max(worst, err)

    return {"rel_err": worst}
