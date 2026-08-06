import importlib.util
import os
import numpy as np


def _run(path):
    spec = importlib.util.spec_from_file_location("learner_regression", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fns = [getattr(mod, n) for n in dir(mod) if n.startswith("test_") and callable(getattr(mod, n))]
    if not fns:
        return None
    for fn in fns:
        fn()
    return True


def _survives(path):
    try:
        return _run(path) is True
    except Exception:
        return False


def check(workdir):
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_unmapped_access": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on correct code: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import batchremap.decode as decode_mod

    good_gather = decode_mod.gather_batch_kv

    def broken_gather(kv_cache, cache_batch_idx, seq_lens):
        cache_batch_idx = np.asarray(cache_batch_idx, dtype=np.int32)
        seq_lens = np.asarray(seq_lens, dtype=np.int32)
        batch_size = len(cache_batch_idx)

        max_len = int(np.max(seq_lens)) if batch_size > 0 else 0
        k_batch = np.zeros((batch_size, max_len, kv_cache.num_heads, kv_cache.head_dim), dtype=kv_cache.k.dtype)
        v_batch = np.zeros((batch_size, max_len, kv_cache.num_heads, kv_cache.head_dim), dtype=kv_cache.v.dtype)

        for i in range(batch_size):
            length = seq_lens[i]
            if length > 0:
                k_batch[i, :length] = kv_cache.k[i, :length]
                v_batch[i, :length] = kv_cache.v[i, :length]

        return k_batch, v_batch

    decode_mod.gather_batch_kv = broken_gather
    import batchremap
    batchremap.gather_batch_kv = broken_gather

    try:
        out["catches_unmapped_access"] = 0.0 if _survives(path) else 1.0
    finally:
        decode_mod.gather_batch_kv = good_gather
        batchremap.gather_batch_kv = good_gather

    return out
