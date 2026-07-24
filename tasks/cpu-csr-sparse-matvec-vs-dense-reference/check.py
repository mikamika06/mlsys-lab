import numpy as np

def _dense_matvec(data, indices, indptr, x):
    """Compute A @ x from its CSR representation using a dense intermediate matrix."""
    n = len(indptr) - 1
    d = x.shape[0]
    # build sparse structure into a dense array (only for reference)
    A = np.zeros((n, d), dtype=np.float64)
    for i in range(n):
        start, end = indptr[i], indptr[i + 1]
        cols = indices[start:end]
        vals = data[start:end]
        A[i, cols] = vals
    return A @ x

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(seed=42)
    cases = []
    for _ in range(5):
        n = rng.integers(10, 30)
        d = rng.integers(10, 30)
        density = rng.random() * 0.3 + 0.05
        # construct CSR
        data_list = []
        indices_list = []
        indptr = [0]
        for _ in range(n):
            mask = rng.random(d) < density
            cols = np.nonzero(mask)[0]
            vals = rng.standard_normal(len(cols))
            data_list.append(vals)
            indices_list.append(cols)
            indptr.append(indptr[-1] + len(cols))
        if not data_list:
            data = np.array([], dtype=np.float64)
            indices = np.array([], dtype=np.int32)
        else:
            data = np.concatenate(data_list)
            indices = np.concatenate(indices_list)
        x = rng.standard_normal(d).astype(np.float64)
        cases.append((data, indices, np.array(indptr, dtype=np.int32), x))

    max_err = 0.0
    for data, indices, indptr, x in cases:
        try:
            y = sol.csr_matvec(data, indices, indptr, x)
        except Exception:
            return {"max_abs_err": float("inf")}
        ref = _dense_matvec(data, indices, indptr, x)
        err = np.max(np.abs(y - ref))
        if err > max_err:
            max_err = err
    return {"max_abs_err": max_err}
