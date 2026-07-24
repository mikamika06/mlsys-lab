import numpy as np

def _ref_pack(a: np.ndarray):
    n = a.shape[0]
    if n % 4 != 0:
        raise ValueError("Length must be divisible by 4")
    blocks = a.reshape(-1, 4)
    abs_blocks = np.abs(blocks)
    # indices of two largest magnitudes per block
    top2_idx = np.argsort(abs_blocks, axis=1)[:, ::-1][:, :2]
    # sort within each block to preserve left‑to‑right order
    sorted_top2 = np.sort(top2_idx, axis=1)
    values = blocks[np.arange(blocks.shape[0])[:, None], sorted_top2].ravel()
    indices = sorted_top2.ravel().astype(np.uint8)
    return values.astype(np.float64), indices

def grade(sol, fx) -> dict:
    # generate a variety of test cases
    cases = [
        np.array([], dtype=np.float64),
        np.arange(4, dtype=np.float64),
        np.array([0.5, -1.2, 0.0, 3.4], dtype=np.float64),
        np.array([2.1, 0.0, -0.7, 0.9], dtype=np.float64),
        np.concatenate([np.arange(8), np.arange(8, 16)]).astype(np.float64),
    ]
    # add random cases
    rng = np.random.default_rng(12345)
    for _ in range(5):
        size = rng.integers(4, 32) // 4 * 4
        cases.append(rng.standard_normal(size))
    for a in cases:
        try:
            values, indices = sol.pack_2_of_4(a)
        except Exception:
            return {"byte_exact_fraction": 0.0}
        if not isinstance(values, np.ndarray) or not isinstance(indices, np.ndarray):
            return {"byte_exact_fraction": 0.0}
        if values.dtype != np.float64 or indices.dtype != np.uint8:
            return {"byte_exact_fraction": 0.0}
        cand_buf = np.concatenate([values.astype(np.float64), indices.astype(np.uint8)]).tobytes()
        ref_vals, ref_idx = _ref_pack(a)
        ref_buf = np.concatenate([ref_vals.astype(np.float64), ref_idx.astype(np.uint8)]).tobytes()
        if cand_buf != ref_buf:
            return {"byte_exact_fraction": 0.0}
    return {"byte_exact_fraction": 1.0}
