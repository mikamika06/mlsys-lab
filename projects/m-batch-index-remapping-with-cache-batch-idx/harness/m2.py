import numpy as np
import ref


def check(workdir):
    from batchremap.cache import KVCacheBuffer
    from batchremap.decode import gather_batch_kv

    out = {"updates_matched": 0.0, "gathers_matched": 0.0}

    scen = ref.generate_cache_test_scenario(seed=123)

    cache = KVCacheBuffer(
        scen["max_cache_batch"],
        scen["max_seq_len"],
        scen["num_heads"],
        scen["head_dim"],
    )

    k_out, v_out = cache.update_and_fetch(
        scen["cache_batch_idx"], scen["seq_lens"], scen["new_k"], scen["new_v"]
    )

    updates_ok = True
    for i, c_idx in enumerate(scen["cache_batch_idx"]):
        pos = scen["seq_lens"][i]
        if not np.allclose(cache.k[c_idx, pos], scen["new_k"][i]):
            updates_ok = False
            out["_note"] = f"K tensor slot {c_idx} at pos {pos} does not match written new_k[{i}]"
            break
        if not np.allclose(cache.v[c_idx, pos], scen["new_v"][i]):
            updates_ok = False
            out["_note"] = f"V tensor slot {c_idx} at pos {pos} does not match written new_v[{i}]"
            break

    if updates_ok:
        out["updates_matched"] = 1.0

    k_batch, v_batch = gather_batch_kv(cache, scen["cache_batch_idx"], scen["seq_lens"] + 1)

    gathers_ok = True
    if k_batch.shape != k_out.shape or v_batch.shape != v_out.shape:
        gathers_ok = False
        out["_note"] = f"gather shapes {k_batch.shape} do not match fetch shapes {k_out.shape}"
    elif not np.allclose(k_batch, k_out) or not np.allclose(v_batch, v_out):
        gathers_ok = False
        out["_note"] = "gather_batch_kv output does not match update_and_fetch gathered outputs"
    else:
        for i, c_idx in enumerate(scen["cache_batch_idx"]):
            length = scen["seq_lens"][i] + 1
            if not np.allclose(k_batch[i, :length], cache.k[c_idx, :length]):
                gathers_ok = False
                out["_note"] = f"gathered history for batch item {i} does not match physical cache at slot {c_idx}"
                break

    if gathers_ok:
        out["gathers_matched"] = 1.0

    return out
