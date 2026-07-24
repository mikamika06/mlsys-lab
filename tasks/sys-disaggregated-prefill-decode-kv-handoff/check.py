import numpy as np

def grade(sol, fx):
    np.random.seed(42)
    L = 3          # layers
    H = 2          # heads
    D = 32         # head dim
    B = 1          # batch
    T = 10         # total length

    # generate full KV cache
    full_cache = []
    for _ in range(L):
        k = np.random.randn(B, H, T, D).astype(np.float32)
        v = np.random.randn(B, H, T, D).astype(np.float32)
        full_cache.append((k, v))

    # split into three chunks (lengths 3,3,4) covering positions 0..9
    lengths = [3, 3, 4]
    starts = [0, 3, 6]
    chunks = []
    for s, length in zip(starts, lengths):
        layer_kv = []
        for layer_idx in range(L):
            k_full, v_full = full_cache[layer_idx]
            layer_kv.append((k_full[:, :, s:s+length, :].copy(),
                             v_full[:, :, s:s+length, :].copy()))
        chunks.append((s, layer_kv))

    # reference: concatenate per layer in order (chunks are already ordered)
    expected = []
    for layer_idx in range(L):
        key_parts = [chunk[1][layer_idx][0] for chunk in chunks]
        val_parts = [chunk[1][layer_idx][1] for chunk in chunks]
        k = np.concatenate(key_parts, axis=2)
        v = np.concatenate(val_parts, axis=2)
        expected.append((k, v))

    # call student
    try:
        result = sol.assemble_cache(chunks)
    except Exception:
        return {"max_abs_err": 1.0}

    # check structure
    if not isinstance(result, list) or len(result) != L:
        return {"max_abs_err": 1.0}
    for i in range(L):
        if not isinstance(result[i], tuple) or len(result[i]) != 2:
            return {"max_abs_err": 1.0}
        k_got, v_got = result[i]
        if not isinstance(k_got, np.ndarray) or not isinstance(v_got, np.ndarray):
            return {"max_abs_err": 1.0}
        if k_got.shape != expected[i][0].shape:
            return {"max_abs_err": 1.0}
        if v_got.shape != expected[i][1].shape:
            return {"max_abs_err": 1.0}

    max_err = 0.0
    for i in range(L):
        err_k = np.max(np.abs(result[i][0].astype(np.float64) - expected[i][0].astype(np.float64)))
        err_v = np.max(np.abs(result[i][1].astype(np.float64) - expected[i][1].astype(np.float64)))
        max_err = max(max_err, err_k, err_v)

    return {"max_abs_err": max_err}
