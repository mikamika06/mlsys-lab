import numpy as np

HEAD_DIM = 64
NUM_HEADS = 8


def _oracle(lens, head_dim, num_heads):
    lens = [int(v) for v in np.asarray(lens).tolist()]
    c = 4 * int(head_dim) * int(num_heads)
    packed = c * sum(n * n for n in lens)
    batch = len(lens)
    max_len = max(lens)
    padded = c * batch * max_len * max_len
    return packed, padded


def grade(sol, fx) -> dict:
    lens = np.asarray(fx["lens"], dtype=np.int64)

    ref_packed, ref_padded = _oracle(lens, HEAD_DIM, NUM_HEADS)

    try:
        got_packed, got_padded = sol.attention_flops(lens.copy(), HEAD_DIM, NUM_HEADS)
        got_packed = int(got_packed)
        got_padded = int(got_padded)
    except Exception:
        return {"exact_match": 0.0}

    ok = 1.0 if (got_packed == ref_packed and got_padded == ref_padded) else 0.0
    return {"exact_match": ok}
