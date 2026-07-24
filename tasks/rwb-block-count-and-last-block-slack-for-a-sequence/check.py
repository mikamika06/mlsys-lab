import numpy as np

def _ref(seqlens, block_size):
    num_blocks = (seqlens + block_size - 1) // block_size
    slack = num_blocks * block_size - seqlens
    return num_blocks.astype(np.int64), slack.astype(np.int64)

def grade(sol, fx) -> dict:
    cases = [
        (np.array([0, 1, 2, 3]), 4),
        (np.array([5, 13, 21, 29, 5]), 8),
        (np.arange(10), 3),
        (np.array([7, 14, 20]), 5),
        (np.array([100, 200, 300]), 50)
    ]
    ok = 1.0
    for seqlens, block_size in cases:
        try:
            got_num, got_slack = sol.block_stats(seqlens, block_size)
            ref_num, ref_slack = _ref(seqlens, block_size)
        except Exception:
            return {"exact_match": 0.0}
        if not (np.array_equal(got_num, ref_num) and np.array_equal(got_slack, ref_slack)):
            ok = 0.0
            break
    return {"exact_match": ok}
