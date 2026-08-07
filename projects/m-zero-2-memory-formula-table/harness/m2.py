import ref
import numpy as np

def check(workdir):
    from zerotwo.comm import calc_bucket_count, toy_reduce_scatter
    max_rel_err = 0.0
    for cfg in ref.BUCKET_TESTS:
        want = ref.ref_calc_bucket_count(cfg["total_elements"], cfg["element_size"], cfg["allgather_bucket_size_bytes"])
        got = calc_bucket_count(cfg["total_elements"], cfg["element_size"], cfg["allgather_bucket_size_bytes"])
        err = abs(float(got) - float(want)) / max(1.0, abs(float(want)))
        if err > max_rel_err:
            max_rel_err = err
    for cfg in ref.REDUCE_SCATTER_TESTS:
        want = ref.ref_toy_reduce_scatter(cfg["grads"], cfg["world_size"])
        got = toy_reduce_scatter(cfg["grads"], cfg["world_size"])
        for w_chunk, g_chunk in zip(want, got):
            w_arr = np.asarray(w_chunk, dtype=np.float32)
            g_arr = np.asarray(g_chunk, dtype=np.float32)
            diff = np.max(np.abs(w_arr - g_arr)) / max(1.0, np.max(np.abs(w_arr)))
            if diff > max_rel_err:
                max_rel_err = float(diff)
    return {"rel_err": float(max_rel_err)}
