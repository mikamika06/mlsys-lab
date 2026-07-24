import numpy as np

from mlsys import scorers

SYM_TOL = 1e-10


def _sym_from_spectrum(rng, w):
    """Build an exactly symmetric matrix with the prescribed eigenvalues."""
    n = len(w)
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    A = q @ np.diag(np.asarray(w, dtype=np.float64)) @ q.T
    return (A + A.T) / 2.0


def _cases():
    rng = np.random.default_rng(0)
    out = []

    # --- symmetric positive definite ---
    out.append(_sym_from_spectrum(rng, [1.0, 2.0, 3.0, 4.0]))
    out.append(_sym_from_spectrum(rng, [1e-3, 0.5, 2.0, 7.0]))
    out.append(_sym_from_spectrum(rng, [5.0, 5.0, 5.0]))
    out.append(np.eye(5))
    out.append(np.array([[4.0, 1.0], [1.0, 3.0]]))
    B = rng.standard_normal((6, 9))
    G = B @ B.T                       # Gram matrix of full-row-rank B -> SPD
    out.append((G + G.T) / 2.0)

    # --- symmetric but NOT positive definite ---
    out.append(_sym_from_spectrum(rng, [3.0, 1.0, -1.0]))          # indefinite
    out.append(_sym_from_spectrum(rng, [2.0, 1.0, 0.5, -1e-6]))    # small negative eigenvalue
    out.append(_sym_from_spectrum(rng, [-1.0, -2.0, -3.0]))        # negative definite
    out.append(np.array([[1.0, 2.0], [2.0, 1.0]]))                 # eigenvalues 3, -1
    out.append(np.zeros((3, 3)))                                   # all eigenvalues 0

    # --- not symmetric (eigenvalues are irrelevant) ---
    out.append(np.array([[2.0, 1.0], [0.0, 3.0]]))
    S = rng.standard_normal((4, 4))
    out.append(_sym_from_spectrum(rng, [1.0, 2.0, 3.0, 4.0]) + 1e-3 * (S - S.T))
    out.append(np.array([[5.0, 0.0, 1.0], [0.0, 5.0, 0.0], [-1.0, 0.0, 5.0]]))

    return out


def _oracle(A):
    """Ground truth from eigenvalue signs, not from Cholesky."""
    A = np.asarray(A, dtype=np.float64)
    if float(np.max(np.abs(A - A.T))) > SYM_TOL:
        return False
    w = np.linalg.eigvalsh((A + A.T) / 2.0)
    return bool(float(np.min(w)) > 0.0)


def _fail(**over):
    out = {"exact_match": 0.0, "factor_max_abs_err": float("inf"), "struct_ok": 0.0}
    out.update(over)
    return out


def grade(sol, fx) -> dict:
    cases = _cases()
    n_ok = 0
    worst = 0.0
    struct_ok = 1.0

    for A in cases:
        want = _oracle(A)

        try:
            got = sol.is_spd(A.copy())
        except Exception:
            return _fail()
        if bool(got) == want:
            n_ok += 1

        try:
            L = sol.cholesky_spd(A.copy())
        except Exception:
            return _fail(exact_match=n_ok / len(cases))

        if not want:
            if L is not None:
                struct_ok = 0.0
            continue

        if L is None:
            struct_ok = 0.0
            worst = float("inf")
            continue

        L = np.asarray(L, dtype=np.float64)
        ref = np.linalg.cholesky(np.asarray(A, dtype=np.float64))
        if L.shape != ref.shape:
            struct_ok = 0.0
            worst = float("inf")
            continue
        if float(np.max(np.abs(np.triu(L, 1)))) > 0.0 or float(np.min(np.diag(L))) <= 0.0:
            struct_ok = 0.0
        worst = max(worst, scorers.max_abs_err(ref, L))

    return {
        "exact_match": float(n_ok) / float(len(cases)),
        "factor_max_abs_err": float(worst),
        "struct_ok": float(struct_ok),
    }
