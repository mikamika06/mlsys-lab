import numpy as np
import ref


def check(workdir):
    from ringattn.ring import ring_attention

    np.random.seed(42)
    q = np.random.randn(1, 2, 4, 8)
    k = np.random.randn(1, 2, 4, 8)
    v = np.random.randn(1, 2, 4, 8)
    steps = 2

    want = ref.ring_attention(q, k, v, steps)
    got = ring_attention(q, k, v, steps)
    err = float(np.max(np.abs(want - got)))
    return {"max_abs_err": err}
