import numpy as np
from mlsys.scorers import max_abs_err

def _build_test_cases():
    """Return a list of (Q, label) pairs generated via NumPy oracle."""
    rng = np.random.RandomState(42)
    cases = []

    # 1. Identity
    cases.append(np.eye(5))

    # 2. Householder reflection (3x3)
    v = np.array([1.0, 1.0, 0.0])
    v = v / np.linalg.norm(v)
    cases.append(np.eye(3) - 2.0 * np.outer(v, v))

    # 3. 2D rotation by pi/7
    theta = np.pi / 7
    c, s = np.cos(theta), np.sin(theta)
    cases.append(np.array([[c, -s], [s, c]]))

    # 4. Random orthogonal from QR (8x8)
    A = rng.randn(8, 8)
    Q_qr, _ = np.linalg.qr(A)
    cases.append(Q_qr)

    # 5. Slightly perturbed orthogonal (8x8)
    cases.append(Q_qr + 1e-6 * rng.randn(8, 8))

    # 6. Non-orthogonal 2x2
    cases.append(np.array([[1.0, 2.0], [3.0, 4.0]]))

    # 7. Product of 10 random Householder reflections (10x10)
    n = 10
    Hbig = np.eye(n)
    for _ in range(n):
        w = rng.randn(n)
        w[0] += np.linalg.norm(w)  # avoid near-zero vector
        w = w / np.linalg.norm(w)
        Hi = np.eye(n) - 2.0 * np.outer(w, w)
        Hbig = Hbig @ Hi
    cases.append(Hbig)

    return cases

def grade(sol, fx) -> dict:
    cases = _build_test_cases()
    worst = 0.0
    for Q in cases:
        n = Q.shape[0]
        expected = float(np.max(np.abs(Q.T @ Q - np.eye(n))))
        try:
            got = float(sol.orthogonality_error(Q))
        except Exception:
            return {"max_abs_err": float("inf")}
        err = max_abs_err(np.array([expected]), np.array([got]))
        worst = max(worst, err)
    return {"max_abs_err": worst}
