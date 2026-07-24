import numpy as np


def lora_rank_r_approx(W: np.ndarray, A0: np.ndarray, B0: np.ndarray) -> dict:
    """Compute the Eckart-Young-optimal rank-r LoRA factorization of a
    full weight delta W, and compare its Frobenius relative
    reconstruction error to a given (possibly non-optimal) rank-r pair.

    W  : (d_out, d_in), a full-rank weight delta (as if from full
         fine-tuning).
    A0, B0 : (d_out, r), (r, d_in), SOME other rank-r factorization to
         compare against (r = A0.shape[1] = B0.shape[0]).

    Via the Eckart-Young theorem, the best rank-r approximation of W in
    Frobenius norm is given by its truncated SVD W = U S V^T, keeping
    the top r singular values: split the singular value symmetrically so
    A @ B reconstructs the same product,
        A = U[:, :r] @ sqrt(diag(S[:r])),
        B = sqrt(diag(S[:r])) @ V^T[:r, :],
    and its relative error is
        sqrt(sum_{i > r} S_i^2) / ||W||_F.

    Returns a dict:
      "A", "B"           : the optimal rank-r factors (d_out, r), (r, d_in).
      "rel_err_optimal"  : Frobenius relative error of A @ B vs W.
      "rel_err_given"    : Frobenius relative error of A0 @ B0 vs W.
    """
    W = np.asarray(W, dtype=np.float64)
    A0 = np.asarray(A0, dtype=np.float64)
    B0 = np.asarray(B0, dtype=np.float64)
    r = A0.shape[1]

    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    sqrt_s = np.sqrt(S[:r])
    A = U[:, :r] * sqrt_s[None, :]
    B = sqrt_s[:, None] * Vt[:r, :]

    norm_W = float(np.linalg.norm(W))
    rel_err_optimal = float(np.linalg.norm(A @ B - W) / norm_W)
    rel_err_given = float(np.linalg.norm(A0 @ B0 - W) / norm_W)

    return {
        "A": A,
        "B": B,
        "rel_err_optimal": rel_err_optimal,
        "rel_err_given": rel_err_given,
    }
