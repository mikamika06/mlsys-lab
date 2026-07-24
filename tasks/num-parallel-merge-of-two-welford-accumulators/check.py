import numpy as np


def _welford(values):
    n = 0
    mean = 0.0
    M2 = 0.0
    for x in np.asarray(values, dtype=np.float64):
        n += 1
        delta = x - mean
        mean += delta / n
        delta2 = x - mean
        M2 += delta * delta2
    return n, mean, M2


def _merge_ref(a, b):
    na, ma, m2a = a
    nb, mb, m2b = b
    if na == 0:
        return nb, mb, m2b
    if nb == 0:
        return na, ma, m2a
    n = na + nb
    delta = mb - ma
    mean = ma + delta * (nb / n)
    M2 = m2a + m2b + delta * delta * (na * nb / n)
    return n, mean, M2


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(12345)
    chunks = [
        rng.normal(size=17),
        rng.normal(loc=1000.0, scale=3.0, size=31),
        rng.normal(loc=-500.0, scale=20.0, size=9),
    ]

    seq = _welford(np.concatenate(chunks))
    merged = (0, 0.0, 0.0)
    for chunk in chunks:
        merged = _merge_ref(merged, _welford(chunk))

    try:
        got = sol.merge_welford(_welford(chunks[0]), _merge_ref(_welford(chunks[1]), _welford(chunks[2])))
        got_arr = np.asarray(got, dtype=np.float64)
        ref_arr = np.asarray(seq, dtype=np.float64)
        err = float(np.linalg.norm(got_arr - ref_arr) / (np.linalg.norm(ref_arr) + 1e-12))
    except Exception:
        err = float("inf")

    return {"rel_err": err}
