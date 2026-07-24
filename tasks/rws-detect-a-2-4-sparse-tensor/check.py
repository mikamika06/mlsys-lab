import numpy as np

def _is_2x4_sparse_ref(tensor):
    arr = np.asarray(tensor)
    if arr.ndim == 0:
        return False
    last = arr.shape[-1]
    if last % 4 != 0:
        return False
    groups = arr.reshape(*arr.shape[:-1], -1, 4)
    nonzeros = np.count_nonzero(groups, axis=-1)
    return bool(np.all(nonzeros == 2))

def grade(sol, fx) -> dict:
    tests = [
        np.array([[1,0,2,0,3,4,0,0],[0,5,0,6,7,0,8,0]]),   # valid
        np.array([[1,0,0,0],[3,4,5,6]]),                    # invalid
        np.arange(12).reshape(2,6),                         # last dim 6 not multiple of 4 -> False
        np.zeros((3,8)),                                    # all zeros -> each block has 0 nonzeros -> False
        np.array([[1,2,3,4,5,6,7,8]]),                      # each block has 4 nonzeros -> False
    ]
    ok = True
    for t in tests:
        try:
            got = sol.is_2x4_sparse(t)
        except Exception:
            return {"exact_match": 0.0}
        expected = _is_2x4_sparse_ref(t)
        if got != expected:
            ok = False
            break
    return {"exact_match": 1.0 if ok else 0.0}
