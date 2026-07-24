import numpy as np

def _reference(A: np.ndarray, k: int):
    vals, vecs = np.linalg.eigh(A)
    idx = np.argsort(vals)[::-1]
    return vals[idx][:k], vecs[:, idx][:, :k]

def grade(sol, fx) -> dict:
    rngs = [np.random.RandomState(0), np.random.RandomState(1), np.random.RandomState(2)]
    ks   = [2, 3, 4]
    max_rel_err = 0.0
    max_align_err = 0.0
    max_ortho_err = 0.0

    for rng, k in zip(rngs, ks):
        n = 6 + k
        A_rand = rng.normal(size=(n, n))
        A_sym = (A_rand + A_rand.T) / 2.0

        try:
            eigvals, eigvecs = sol.topk_deflation(A_sym, k)
        except Exception:
            return {"rel_err_eig": float("inf"),
                    "vec_align_err": float("inf"),
                    "orthogonality_err": float("inf")}

        ref_vals, ref_vecs = _reference(A_sym, k)

        if eigvals.shape != (k,) or eigvecs.shape != (n, k):
            return {"rel_err_eig": float("inf"),
                    "vec_align_err": float("inf"),
                    "orthogonality_err": float("inf")}

        # eigenvalue relative error
        rel_err = np.max(np.abs(eigvals - ref_vals)) / (np.max(ref_vals) + 1e-12)
        max_rel_err = max(max_rel_err, rel_err)

        # vector alignment error
        align_err = 0.0
        for i in range(k):
            v = eigvecs[:, i]
            best_j = None
            best_dot = -1.0
            for j in range(k):
                dot = np.abs(v @ ref_vecs[:, j])
                if dot > best_dot:
                    best_dot = dot
                    best_j = j
            # align sign
            if (v @ ref_vecs[:, best_j]) < 0:
                v = -v
            err = np.linalg.norm(v - ref_vecs[:, best_j]) / np.linalg.norm(ref_vecs[:, best_j])
            if err > align_err:
                align_err = err
        max_align_err = max(max_align_err, align_err)

        # orthogonality error
        ortho_err = np.linalg.norm(eigvecs.T @ eigvecs - np.eye(k))
        max_ortho_err = max(max_ortho_err, ortho_err)

    return {"rel_err_eig": max_rel_err,
            "vec_align_err": max_align_err,
            "orthogonality_err": max_ortho_err}
