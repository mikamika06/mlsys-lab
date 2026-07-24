import numpy as np

def grade(sol, fx) -> dict:
    cases = [
        (np.array([5, 12]), 4),
        (np.array([8, 16, 3]), 8),
        (np.array([0, 1, 2]), 2),
        (np.array([7]), 3),
        (np.arange(10) + 1, 5)
    ]
    ok = 1.0
    for seq_lengths, block_size in cases:
        try:
            got = sol.mark_kv_blocks(seq_lengths, block_size)
            if not isinstance(got, np.ndarray):
                ok = 0.0
                break
            sl = np.asarray(seq_lengths, dtype=int)
            n_blocks_per_seq = (sl + block_size - 1) // block_size
            total_blocks = int(n_blocks_per_seq.sum())
            ref = np.empty(total_blocks, dtype=bool)
            idx = 0
            for L, nb in zip(sl, n_blocks_per_seq):
                full_blocks = nb if L % block_size == 0 else nb - 1
                ref[idx:idx+full_blocks] = True
                idx += full_blocks
                if nb > full_blocks:
                    ref[idx] = False
                    idx += 1
            if not np.array_equal(got, ref):
                ok = 0.0
                break
        except Exception:
            ok = 0.0
            break
    return {"exact_match": ok}
