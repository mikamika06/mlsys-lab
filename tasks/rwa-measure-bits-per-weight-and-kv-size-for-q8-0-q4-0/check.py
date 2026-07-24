import numpy as np
from mlsys import scorers

def grade(sol, fx) -> dict:
    try:
        got = sol.measure_qkv()
        got_arr = np.array(got, dtype=np.float64)
    except Exception:
        return {"rel_err": 0.0}
    block_size = 32
    bpw_q8 = (34 * 8) / block_size
    bpw_q4 = (18 * 8) / block_size
    kv_ratio_q8 = 34 / (block_size * 2)
    kv_ratio_q4 = 18 / (block_size * 2)
    ref_arr = np.array([bpw_q8, bpw_q4, kv_ratio_q8, kv_ratio_q4], dtype=np.float64)
    err = scorers.rel_err(ref_arr, got_arr)
    return {"rel_err": err}
