import numpy as np
import ref


def check(workdir):
    from ringattn.softmax import online_update
    q, k, v = ref.get_test_cases()
    scale = 1.0 / np.sqrt(q.shape[-1])

    k_chunks = np.array_split(k, 4, axis=1)
    v_chunks = np.array_split(v, 4, axis=1)

    m = np.full((q.shape[0], q.shape[1], 1), -np.inf, dtype=np.float32)
    l = np.zeros((q.shape[0], q.shape[1], 1), dtype=np.float32)
    o = np.zeros_like(q, dtype=np.float32)

    for kc, vc in zip(k_chunks, v_chunks):
        scores = np.matmul(q, np.swapaxes(kc, -1, -2)) * scale
        m, l, o = online_update(m, l, o, scores, vc)

    want = ref.naive_full_attention(q, k, v)
    err = float(np.max(np.abs(o - want)))
    return {"max_abs_err": err}
