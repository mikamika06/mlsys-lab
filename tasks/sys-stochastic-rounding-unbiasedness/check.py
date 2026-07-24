import numpy as np


def _oracle_stochastic_round(x, rng):
    x = np.asarray(x, dtype=np.float32)
    nearest16 = x.astype(np.float16)
    nearest = nearest16.astype(np.float32)

    lower16 = np.where(
        nearest <= x,
        nearest16,
        np.nextafter(nearest16, np.float16(-np.inf)),
    )
    upper16 = np.where(
        nearest <= x,
        np.nextafter(nearest16, np.float16(np.inf)),
        nearest16,
    )

    lower = lower16.astype(np.float32)
    upper = upper16.astype(np.float32)

    same = lower == upper
    prob = np.zeros_like(x, dtype=np.float32)
    prob[~same] = (x[~same] - lower[~same]) / (upper[~same] - lower[~same])

    choose_upper = rng.random(x.shape) < prob
    return np.where(choose_upper, upper, lower).astype(np.float32)


def grade(sol, fx) -> dict:
    base = np.array(
        [
            1.0e-5,
            1.0e-5,
            1.0e-5,
            1.0e-5,
            1.0e-5,
        ],
        dtype=np.float32,
    )
    x16 = base.astype(np.float16)
    x = (
        x16.astype(np.float32)
        + np.nextafter(x16, np.float16(np.inf)).astype(np.float32)
    ) / 2.0

    ref_rng = np.random.default_rng(12345)
    ref_samples = []
    for _ in range(20000):
        ref_samples.append(_oracle_stochastic_round(x, ref_rng))
    ref_mean = np.mean(np.stack(ref_samples), axis=0)

    cand_rng = np.random.default_rng(12345)
    cand_samples = []
    try:
        for _ in range(20000):
            cand_samples.append(
                np.asarray(sol.stochastic_round(x, cand_rng), dtype=np.float32)
            )
        cand_mean = np.mean(np.stack(cand_samples), axis=0)
    except Exception:
        return {"rel_err": float("inf")}

    err = np.linalg.norm(cand_mean - ref_mean)
    denom = np.linalg.norm(ref_mean) + 1e-12
    return {"rel_err": float(err / denom)}
