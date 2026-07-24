import numpy as np

from mlsys import scorers


def _oracle(W, A0, B0):
    r = A0.shape[1]
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    sqrt_s = np.sqrt(S[:r])
    A = U[:, :r] * sqrt_s[None, :]
    B = sqrt_s[:, None] * Vt[:r, :]

    norm_W = float(np.linalg.norm(W))
    rel_err_optimal = float(np.linalg.norm(A @ B - W) / norm_W)
    rel_err_given = float(np.linalg.norm(A0 @ B0 - W) / norm_W)
    return rel_err_optimal, rel_err_given


def _synthetic_cases():
    rng = np.random.default_rng(83)
    cases = []
    for _ in range(4):
        d_out = int(rng.integers(10, 30))
        d_in = int(rng.integers(8, 25))
        r = int(rng.integers(1, min(d_out, d_in) // 2 + 1))
        W = rng.standard_normal((d_out, d_in)) * rng.uniform(0.5, 3.0)
        A0 = rng.standard_normal((d_out, r))
        B0 = rng.standard_normal((r, d_in))
        cases.append((W, A0, B0))
    return cases


def grade(sol, fx) -> dict:
    cases = [(fx["W"], fx["A0"], fx["B0"])] + _synthetic_cases()

    worst_rel_err = 0.0
    ok_count = 0
    total = 0

    for W, A0, B0 in cases:
        ref_opt, ref_given = _oracle(W, A0, B0)
        norm_W = float(np.linalg.norm(W))
        total += 1

        try:
            got = sol.lora_rank_r_approx(W.copy(), A0.copy(), B0.copy())
            got_opt = float(got["rel_err_optimal"])
            got_given = float(got["rel_err_given"])
            A = np.asarray(got["A"], dtype=np.float64)
            B = np.asarray(got["B"], dtype=np.float64)
            actual_recon_err = float(np.linalg.norm(A @ B - W) / norm_W)
        except Exception:
            worst_rel_err = float("inf")
            continue

        ref_vec = np.array([ref_opt, ref_given, ref_opt])
        got_vec = np.array([got_opt, got_given, actual_recon_err])
        err = scorers.rel_err(ref_vec, got_vec)
        worst_rel_err = max(worst_rel_err, err)

        if got_opt >= ref_opt - 1e-9:
            ok_count += 1

    return {
        "rel_err": worst_rel_err,
        "not_below_bound": (ok_count / total) if total else 0.0,
    }
