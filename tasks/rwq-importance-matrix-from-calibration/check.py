import numpy as np

from mlsys import scorers


def _oracle_imatrix(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return np.sum(X ** 2, axis=0)


def grade(sol, fx) -> dict:
    errs = []

    # fixture: realistic calibration activations with a few "hot" channels
    X = np.asarray(fx["gguf_x"], dtype=np.float64)
    expected = _oracle_imatrix(X)
    try:
        got = np.asarray(sol.imatrix_from_calibration(X), dtype=np.float64)
        errs.append(
            scorers.rel_err(expected, got) if got.shape == expected.shape else float("inf")
        )
    except Exception:
        errs.append(float("inf"))

    # a second, independently generated calibration batch (different shape and
    # scale) so a solution can't pass by overfitting to the fixture's size
    rng = np.random.default_rng(1)
    for n_tokens, n_channels in [(300, 40), (17, 5)]:
        X2 = rng.standard_normal((n_tokens, n_channels)) * rng.uniform(0.1, 5.0, size=n_channels)
        expected2 = _oracle_imatrix(X2)
        try:
            got2 = np.asarray(sol.imatrix_from_calibration(X2), dtype=np.float64)
            errs.append(
                scorers.rel_err(expected2, got2) if got2.shape == expected2.shape else float("inf")
            )
        except Exception:
            errs.append(float("inf"))

    return {"rel_err": max(errs)}
