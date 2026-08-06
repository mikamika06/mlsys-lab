import math
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

    d_out = W.shape[0]
    d_in = W.shape[1]
    W_curr = np.array(W, dtype=np.float64)

    U_list = []
    S_list = []
    Vt_list = []

    for i in range(r):
        v = np.zeros(d_in, dtype=np.float64)
        for j in range(d_in):
            v[j] = 1.0 / math.sqrt(d_in) + 0.1 * (j % 5)

        norm_v = 0.0
        for j in range(d_in):
            norm_v += v[j] * v[j]
        norm_v = math.sqrt(norm_v)
        for j in range(d_in):
            v[j] /= norm_v

        for _ in range(150):
            u = np.zeros(d_out, dtype=np.float64)
            for r_idx in range(d_out):
                s_val = 0.0
                for c_idx in range(d_in):
                    s_val += W_curr[r_idx, c_idx] * v[c_idx]
                u[r_idx] = s_val

            norm_u = 0.0
            for r_idx in range(d_out):
                norm_u += u[r_idx] * u[r_idx]
            norm_u = math.sqrt(norm_u)
            if norm_u == 0.0:
                break
            for r_idx in range(d_out):
                u[r_idx] /= norm_u

            v_new = np.zeros(d_in, dtype=np.float64)
            for c_idx in range(d_in):
                s_val = 0.0
                for r_idx in range(d_out):
                    s_val += W_curr[r_idx, c_idx] * u[r_idx]
                v_new[c_idx] = s_val

            norm_vnew = 0.0
            for c_idx in range(d_in):
                norm_vnew += v_new[c_idx] * v_new[c_idx]
            norm_vnew = math.sqrt(norm_vnew)
            if norm_vnew == 0.0:
                break
            for c_idx in range(d_in):
                v_new[c_idx] /= norm_vnew

            v = v_new

        sigma = 0.0
        for r_idx in range(d_out):
            row_dot = 0.0
            for c_idx in range(d_in):
                row_dot += W_curr[r_idx, c_idx] * v[c_idx]
            sigma += u[r_idx] * row_dot

        U_list.append(u)
        S_list.append(sigma)
        Vt_list.append(v)

        for r_idx in range(d_out):
            for c_idx in range(d_in):
                W_curr[r_idx, c_idx] -= sigma * u[r_idx] * v[c_idx]

    A = np.zeros((d_out, r), dtype=np.float64)
    B = np.zeros((r, d_in), dtype=np.float64)

    for k in range(r):
        sqrt_s = math.sqrt(S_list[k])
        u_k = U_list[k]
        v_k = Vt_list[k]
        for i in range(d_out):
            A[i, k] = u_k[i] * sqrt_s
        for j in range(d_in):
            B[k, j] = sqrt_s * v_k[j]

    sum_sq_W = 0.0
    for i in range(d_out):
        for j in range(d_in):
            val = W[i, j]
            sum_sq_W += val * val
    norm_W = math.sqrt(sum_sq_W)

    AB = np.zeros((d_out, d_in), dtype=np.float64)
    for i in range(d_out):
        for j in range(d_in):
            s_val = 0.0
            for k in range(r):
                s_val += A[i, k] * B[k, j]
            AB[i, j] = s_val

    sum_sq_diff_opt = 0.0
    for i in range(d_out):
        for j in range(d_in):
            diff = AB[i, j] - W[i, j]
            sum_sq_diff_opt += diff * diff
    rel_err_optimal = float(math.sqrt(sum_sq_diff_opt) / norm_W)

    A0B0 = np.zeros((d_out, d_in), dtype=np.float64)
    for i in range(d_out):
        for j in range(d_in):
            s_val = 0.0
            for k in range(r):
                s_val += A0[i, k] * B0[k, j]
            A0B0[i, j] = s_val

    sum_sq_diff_given = 0.0
    for i in range(d_out):
        for j in range(d_in):
            diff = A0B0[i, j] - W[i, j]
            sum_sq_diff_given += diff * diff
    rel_err_given = float(math.sqrt(sum_sq_diff_given) / norm_W)

    return {
        "A": A,
        "B": B,
        "rel_err_optimal": rel_err_optimal,
        "rel_err_given": rel_err_given,
    }
