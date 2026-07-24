import numpy as np

def _oracle(mask, cu_seqlens):
    N = mask.shape[0]
    seg_ids = np.empty(N, dtype=np.int32)
    for i in range(len(cu_seqlens)-1):
        seg_ids[cu_seqlens[i]:cu_seqlens[i+1]] = i
    return bool(np.any((seg_ids[:,None] != seg_ids[None,:]) & mask.astype(bool)))

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    ok = 1.0
    for _ in range(20):
        B = rng.integers(2,6)
        seq_lens = rng.integers(1,5,size=B)
        cu_seqlens = np.concatenate([[0], np.cumsum(seq_lens)])
        N = cu_seqlens[-1]
        mask = rng.integers(0,2,size=(N,N))
        try:
            got = sol.detect_leakage(mask, cu_seqlens)
        except Exception:
            return {"exact_match": 0.0}
        exp = _oracle(mask, cu_seqlens)
        if bool(got) != exp:
            ok = 0.0
            break
    return {"exact_match": ok}
