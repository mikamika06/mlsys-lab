import numpy as np


def _oracle_attention_ai(seqlen, dim):
    n = int(seqlen)
    d = int(dim)
    flops_qk = 2.0 * n * n * d
    bytes_qk = (2.0 * n * d + n * n) * 4.0
    ai_qk = flops_qk / bytes_qk

    flops_softmax = 5.0 * n * n
    bytes_softmax = (2.0 * n * n) * 4.0
    ai_softmax = flops_softmax / bytes_softmax

    flops_pv = 2.0 * n * n * d
    bytes_pv = (n * n + n * d) * 4.0
    ai_pv = flops_pv / bytes_pv

    return np.asarray([ai_qk, ai_softmax, ai_pv], dtype=np.float64)


def grade(sol, fx) -> dict:
    cases = [
        (128, 32),
        (512, 64),
        (1024, 128),
        (2048, 256),
        (77, 16),
    ]

    max_err = 0.0
    for seqlen, dim in cases:
        try:
            got = np.asarray(sol.attention_ai(seqlen, dim), dtype=np.float64)
        except Exception:
            return {"rel_err": 1.0}

        ref = _oracle_attention_ai(seqlen, dim)
        if got.shape != ref.shape:
            return {"rel_err": 1.0}

        err = np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-12)
        max_err = max(max_err, float(err))

    return {"rel_err": max_err}
