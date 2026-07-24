def grade(sol, fx) -> dict:
    import math
    import numpy as np

    cases = [
        (5, 3),
        (10, 4),
        (7, 2),
        (0, 5),   # edge case: empty sequence
        (13, 1)
    ]

    ok = 1.0
    for seq_len, block_size in cases:
        try:
            got = sol.measure_blocks(seq_len, block_size)
            if not isinstance(got, tuple) or len(got) != 3:
                ok = 0.0
                break
            num_blocks, slot_mapping, waste = got

            # Reference calculation
            ref_num_blocks = -(-seq_len // block_size)          # ceil division
            ref_waste = ref_num_blocks * block_size - seq_len
            ref_slot_mapping = np.arange(ref_num_blocks * block_size,
                                         dtype=np.int64)[:seq_len]

            if num_blocks != ref_num_blocks or waste != ref_waste:
                ok = 0.0
                break

            if not np.array_equal(slot_mapping, ref_slot_mapping):
                ok = 0.0
                break
        except Exception:
            ok = 0.0
            break

    return {"exact_match": ok}
