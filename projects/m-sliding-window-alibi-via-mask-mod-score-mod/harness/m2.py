import numpy as np
import ref

def check(workdir):
    out = {"sparsity_err": 1e9, "nums_match": 0.0, "idxs_match": 0.0}
    try:
        from flex.block_mask import compute_block_mask_indices
    except ImportError:
        out["_note"] = "failed to import compute_block_mask_indices"
        return out

    try:
        g_s, g_n, g_i = compute_block_mask_indices(1024, 256, 128)
        w_s, w_n, w_i = ref.compute_block_mask_indices(1024, 256, 128)

        out["sparsity_err"] = float(abs(g_s - w_s))

        if np.array_equal(g_n, w_n):
            out["nums_match"] = 1.0
        else:
            out["_note_nums"] = "kv_num_blocks mismatch"

        if np.array_equal(g_i, w_i):
            out["idxs_match"] = 1.0
        else:
            out["_note_idxs"] = "kv_indices mismatch"

        if g_i.shape != w_i.shape:
            out["_note_shape"] = f"idx shape mismatch: got {g_i.shape}, want {w_i.shape}"

    except Exception as e:
        out["_note"] = f"error: {e}"

    return out
