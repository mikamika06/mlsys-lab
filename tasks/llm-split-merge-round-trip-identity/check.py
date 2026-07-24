import numpy as np
from mlsys import scorers

def _ref_split_merge(x, num_heads):
    batch, seq_len, dim = x.shape
    head_dim = dim // num_heads
    split = x.reshape(batch, seq_len, num_heads, head_dim)
    merged = split.reshape(batch, seq_len, dim)
    return merged

def grade(sol, fx) -> dict:
    max_err = 0.0
    # test cases with varying shapes and head counts
    for batch in [1, 2, 3]:
        for seq_len in [4, 5]:
            for num_heads in [2, 4, 8]:
                dim = num_heads * 8  # ensure divisible
                x = np.random.randn(batch, seq_len, dim).astype(np.float64)
                try:
                    split = sol.split_heads(x, num_heads)
                    merged = sol.merge_heads(split)
                except Exception:
                    return {"max_abs_err": float("inf")}
                ref_merged = _ref_split_merge(x, num_heads)
                err = scorers.max_abs_err(ref_merged, merged)
                if err > max_err:
                    max_err = err
    return {"max_abs_err": max_err}
