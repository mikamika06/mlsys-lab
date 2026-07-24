import numpy as np

def _ref(block_size, seq_lengths):
    arr = np.array(seq_lengths, dtype=np.int64)
    blocks = (arr + block_size - 1) // block_size
    wasted = blocks * block_size - arr
    return list(blocks), list(wasted)

def grade(sol, fx) -> dict:
    cases = [
        (4, [5, 8, 3]),
        (7, [0, 1, 6, 14]),
        (10, [9, 20, 31]),
        (3, [2, 3, 4, 5]),
        (12, list(range(0, 25))),
    ]
    ok = 1.0
    for bs, seqs in cases:
        try:
            got_blocks, got_wasted = sol.compute_block_and_waste(bs, seqs)
        except Exception:
            return {"exact_match": 0.0}
        ref_blocks, ref_wasted = _ref(bs, seqs)
        if got_blocks != ref_blocks or got_wasted != ref_wasted:
            return {"exact_match": 0.0}
    return {"exact_match": ok}
