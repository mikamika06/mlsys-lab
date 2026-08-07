import numpy as np
import ref


def check(workdir):
    from ringattn.simulate import ring_attention_simulate

    np.random.seed(123)
    q = np.random.randn(64, 32)
    k = np.random.randn(64, 32)
    v = np.random.randn(64, 32)
    num_ranks = 4

    want = ref.ring_attention_simulate(q, k, v, num_ranks)
    got = ring_attention_simulate(q, k, v, num_ranks)

    diff = np.max(np.abs(want - got))
    return {"rel_err": float(diff)}
