import math
import numpy as np

def pca_power_iteration(X: np.ndarray, k: int) -> np.ndarray:
    """
    Compute the first `k` principal components of a centered data matrix `X`
    using power iteration with deflation.  The algorithm is deterministic
    because it uses a fixed random seed.
    """
    rng = np.random.default_rng(0)
    n = X.shape[0]
    d = X.shape[1]

    C = np.zeros((d, d), dtype=X.dtype)
    for i in range(d):
        for j in range(d):
            s = 0.0
            for m in range(n):
                s += float(X[m, i]) * float(X[m, j])
            C[i, j] = s

    comps: list[np.ndarray] = []

    for _ in range(k):
        v = rng.standard_normal(d)
        if comps:
            for u in comps:
                dot_uv = 0.0
                for i in range(d):
                    dot_uv += float(u[i]) * float(v[i])
                for i in range(d):
                    v[i] -= dot_uv * float(u[i])

        sum_v2 = 0.0
        for i in range(d):
            sum_v2 += float(v[i]) * float(v[i])
        norm_v = math.sqrt(sum_v2)

        if norm_v == 0.0:
            v = rng.standard_normals(d)
            sum_v2 = 0.0
            for i in range(d):
                sum_v2 += float(v[i]) * float(v[i])
            norm_v = math.sqrt(sum_v2)

        for i in range(d):
            v[i] /= norm_v

        for _ in range(2000):
            w = np.zeros(d, dtype=X.dtype)
            for i in range(d):
                s = 0.0
                for j in range(d):
                    s += float(C[i, j]) * float(v[j])
                w[i] = s

            if comps:
                for u in comps:
                    dot_uw = 0.0
                    for i in range(d):
                        dot_uw += float(u[i]) * float(w[i])
                    for i in range(d):
                        w[i] -= dot_uw * float(u[i])

            sum_w2 = 0.0
            for i in range(d):
                sum_w2 += float(w[i]) * float(w[i])
            norm_w = math.sqrt(sum_w2)

            if norm_w == 0.0:
                break

            v_new = np.zeros(d, dtype=X.dtype)
            for i in range(d):
                v_new[i] = w[i] / norm_w

            all_close = True
            for i in range(d):
                diff = abs(float(v[i]) - float(v_new[i]))
                limit = 1e-10 + 1e-5 * abs(float(v_new[i]))
                if diff > limit:
                    all_close = False
                    break

            if all_close:
                v = v_new
                break
            v = v_new

        comps.append(v.copy())

    return np.stack(comps)
