import ref
import numpy as np

def check(workdir):
    from checkpoint.merge import merge_tp_shards
    shards, axis_map = ref.generate_shards(4)
    want = ref.merge_tp_shards(shards, axis_map)
    try:
        got = merge_tp_shards(shards, axis_map)
    except Exception:
        return {"byte_exact_fraction": 0.0}

    matched_bytes = 0
    total_bytes = 0
    for k, v_want in want.items():
        b = v_want.nbytes
        total_bytes += b
        if k in got and got[k].shape == v_want.shape and np.array_equal(got[k], v_want):
            matched_bytes += b

    return {"byte_exact_fraction": matched_bytes / max(1, total_bytes)}
