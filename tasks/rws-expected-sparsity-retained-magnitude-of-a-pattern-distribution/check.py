import itertools

import numpy as np

from mlsys import scorers


def _pattern_masks() -> np.ndarray:
    masks = np.zeros((6, 4), dtype=np.float64)
    for k, (i, j) in enumerate(itertools.combinations(range(4), 2)):
        masks[k, i] = 1.0
        masks[k, j] = 1.0
    return masks


_MASKS = _pattern_masks()


def _oracle(p: np.ndarray, w: np.ndarray):
    p = np.asarray(p, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    density = p @ (_MASKS.sum(axis=1) / 4.0)          # (G,)
    marginal_keep = p @ _MASKS                        # (G,4)
    retained = np.sum(marginal_keep * w, axis=1)       # (G,)
    return density, retained


def _cases(rng: np.random.Generator):
    cases = []

    # hand-checkable single group
    p0 = np.array([[0.5, 0.5, 0.0, 0.0, 0.0, 0.0]])
    w0 = np.array([[1.0, 2.0, 3.0, 4.0]])
    cases.append((p0, w0))

    # deterministic one-hot patterns
    p1 = np.eye(6)
    w1 = rng.uniform(0.1, 5.0, size=(6, 4))
    cases.append((p1, w1))

    # random batch, different G from the fixture
    G = 23
    alpha = rng.uniform(0.2, 4.0, size=6)
    p2 = rng.dirichlet(alpha, size=G)
    w2 = np.abs(rng.standard_normal((G, 4))) * rng.uniform(0.05, 3.0, size=(G, 1))
    cases.append((p2, w2))

    return cases


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(1)
    cases = [(fx["pat_p"], fx["pat_w"])] + _cases(rng)

    errs = []
    for p, w in cases:
        exp_density, exp_retained = _oracle(p, w)
        try:
            got_density, got_retained = sol.expected_pattern_stats(
                np.array(p, copy=True), np.array(w, copy=True)
            )
            got_density = np.asarray(got_density, dtype=np.float64)
            got_retained = np.asarray(got_retained, dtype=np.float64)
        except Exception:
            errs.append(float("inf"))
            continue

        if got_density.shape != exp_density.shape or got_retained.shape != exp_retained.shape:
            errs.append(float("inf"))
            continue

        exp_cat = np.concatenate([exp_density, exp_retained])
        got_cat = np.concatenate([got_density, got_retained])
        errs.append(scorers.rel_err(exp_cat, got_cat))

    return {"rel_err": max(errs) if errs else float("inf")}
