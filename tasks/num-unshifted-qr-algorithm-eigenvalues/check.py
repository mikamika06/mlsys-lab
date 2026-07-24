import numpy as np


def _oracle(A):
    return np.sort(np.linalg.eigvalsh(A).astype(np.float64))


def _rel_err(a, b):
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)) /
                 (np.linalg.norm(np.asarray(b, dtype=np.float64)) + 1e-12))


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(12345)
    cases = []
    for n in [2, 3, 5, 8]:
        M = rng.normal(size=(n, n))
        cases.append((M + M.T) / 2.0)
    cases.append(np.array([
        [4.0, 1.0, 0.5],
        [1.0, 3.0, -0.2],
        [0.5, -0.2, 2.0],
    ]))

    score = 1.0
    worst = 0.0
    for A in cases:
        ref = _oracle(A)
        try:
            got = np.sort(np.asarray(sol.qr_eigenvalues(A.copy()), dtype=np.float64))
            err = _rel_err(got, ref)
            worst = max(worst, err)
            if err >= 1e-6 or got.shape != ref.shape:
                score = 0.0
                break
        except Exception:
            score = 0.0
            break
    return {"rel_err": worst if score else 1.0}
