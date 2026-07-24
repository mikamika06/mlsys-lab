import numpy as np


def _oracle_top_salient(X: np.ndarray, frac: float) -> set:
    X = np.asarray(X, dtype=np.float64)
    C = X.shape[1]
    scores = np.mean(np.abs(X), axis=0)
    k = max(1, int(np.ceil(frac * C)))
    order = np.argsort(-scores, kind="stable")
    return set(int(i) for i in order[:k])


def _cases(rng: np.random.Generator):
    cases = []

    # small, hand-checkable
    X0 = np.array([
        [1.0, 0.1, -0.2],
        [-1.0, 0.2, 0.1],
        [0.9, -0.1, 0.0],
    ])
    cases.append((X0, 0.34))

    # wider batch, default frac
    n, C = 300, 150
    Xb = rng.standard_normal((n, C)) * rng.uniform(0.05, 0.3, size=C)
    hot = rng.choice(C, size=3, replace=False)
    Xb[:, hot] *= rng.uniform(6.0, 15.0, size=3)
    cases.append((Xb, 0.01))

    # a different frac
    n2, C2 = 120, 64
    Xc = rng.standard_normal((n2, C2)) * rng.uniform(0.1, 0.5, size=C2)
    hot2 = rng.choice(C2, size=5, replace=False)
    Xc[:, hot2] *= rng.uniform(5.0, 10.0, size=5)
    cases.append((Xc, 0.05))

    return cases


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    cases = [(np.asarray(fx["awq_x"], dtype=np.float64), 0.01)] + _cases(rng)

    hits = 0
    for X, frac in cases:
        expected = _oracle_top_salient(X, frac)
        try:
            got = sol.top_salient_channels(np.array(X, copy=True), frac)
            got_set = set(int(i) for i in np.asarray(got).ravel())
        except Exception:
            continue

        if got_set == expected:
            hits += 1

    return {"exact_match": hits / len(cases)}
