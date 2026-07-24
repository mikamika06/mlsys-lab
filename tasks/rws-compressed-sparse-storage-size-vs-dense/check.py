import math
import numpy as np

def _ref(tensor):
    """Oracle: compute compressed-sparse footprint from scratch."""
    n = len(tensor)
    nnz = int(np.count_nonzero(tensor))
    dense_bytes = int(tensor.nbytes)
    sparse_bytes = nnz * 2 + math.ceil(n / 8)
    if sparse_bytes == 0:
        ratio = float("inf")
    else:
        ratio = dense_bytes / sparse_bytes
    return sparse_bytes, dense_bytes, ratio

def grade(sol, fx) -> dict:
    np.random.seed(42)

    # Case 1: moderate pruning (~50 %)
    t1 = np.random.randn(1000).astype(np.float16)
    t1[::2] = 0

    # Case 2: heavy pruning (~90 %)
    t2 = np.random.randn(10000).astype(np.float16)
    t2[np.random.rand(10000) > 0.1] = 0

    # Case 3: no pruning
    t3 = np.random.randn(100).astype(np.float16)

    # Case 4: total pruning
    t4 = np.zeros(500, dtype=np.float16)

    # Case 5: single element
    t5 = np.array([1.0], dtype=np.float16)

    # Case 6: non-byte-aligned n
    t6 = np.random.randn(13).astype(np.float16)
    t6[0] = 0
    t6[3] = 0
    t6[7] = 0

    cases = [t1, t2, t3, t4, t5, t6]

    for tensor in cases:
        ref_sparse, ref_dense, ref_ratio = _ref(tensor)
        try:
            got = sol.compressed_sparse_footprint(tensor.copy())
            got_sparse, got_dense, got_ratio = int(got[0]), int(got[1]), float(got[2])
        except Exception:
            return {"size_ratio": 0.0}

        if got_sparse != ref_sparse:
            return {"size_ratio": 0.0}
        if got_dense != ref_dense:
            return {"size_ratio": 0.0}
        if abs(got_ratio - ref_ratio) > 1e-6:
            return {"size_ratio": 0.0}

    return {"size_ratio": 1.0}
