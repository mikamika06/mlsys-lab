import math
import numpy as np


def _ref_classify(mask_mod, seq_len_q, seq_len_kv, block_q, block_kv):
    num_q = seq_len_q // block_q
    num_kv = seq_len_kv // block_kv
    result = []
    for i in range(num_q):
        row = []
        for j in range(num_kv):
            vals = []
            for q in range(i * block_q, (i + 1) * block_q):
                for k in range(j * block_kv, (j + 1) * block_kv):
                    vals.append(bool(mask_mod(0, 0, q, k)))
            if all(vals):
                row.append("full")
            elif not any(vals):
                row.append("empty")
            else:
                row.append("partial")
        result.append(row)
    return result


def grade(sol, fx) -> dict:
    def causal(b, h, q, k):
        return k <= q

    def sliding_window(b, h, q, k):
        return abs(q - k) <= 2

    def full_mask(b, h, q, k):
        return True

    def empty_mask(b, h, q, k):
        return False

    test_cases = [
        (causal, 8, 8, 2, 2),
        (causal, 4, 4, 2, 2),
        (sliding_window, 8, 8, 2, 2),
        (full_mask, 6, 6, 2, 2),
        (empty_mask, 6, 6, 3, 3),
        (causal, 8, 8, 4, 4),
    ]

    ok = 1.0
    for mask_mod, sq, skv, bq, bkv in test_cases:
        ref = _ref_classify(mask_mod, sq, skv, bq, bkv)
        try:
            got = sol.classify_block_mask_tiles(mask_mod, sq, skv, bq, bkv)
            got = [list(row) for row in got]
        except Exception:
            ok = 0.0
            break
        if got != ref:
            ok = 0.0
            break

    return {"exact_match": ok}
