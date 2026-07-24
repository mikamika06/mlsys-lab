import numpy as np


def _oracle(token_ids, embedding, output_weight, world_size):
    vocab, hidden = embedding.shape
    n = token_ids.shape[0]

    gathered = np.zeros((n, hidden), dtype=np.float64)
    logits = np.zeros((n, vocab), dtype=np.float64)

    bounds = np.linspace(0, vocab, world_size + 1, dtype=int)
    for rank in range(world_size):
        start = bounds[rank]
        end = bounds[rank + 1]

        local_hidden = np.zeros((n, hidden), dtype=np.float64)
        mask = (token_ids >= start) & (token_ids < end)
        local_hidden[mask] = embedding[token_ids[mask]]
        gathered += local_hidden

        local_logits = np.zeros((n, vocab), dtype=np.float64)
        local_logits[:, start:end] = gathered @ output_weight[start:end].T
        logits += local_logits

    return gathered, logits


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([0, 3, 5, 7]),
            np.arange(40, dtype=np.float64).reshape(10, 4) / 7.0,
            3,
        ),
        (
            np.array([9, 1, 4, 4, 0]),
            np.sin(np.arange(54, dtype=np.float64).reshape(9, 6)),
            4,
        ),
        (
            np.array([2, 6, 8]),
            np.cos(np.arange(45, dtype=np.float64).reshape(9, 5)),
            2,
        ),
    ]

    err = 0.0
    for tokens, emb, world in cases:
        out = emb * 0.37 + 0.11
        ref_h, ref_l = _oracle(tokens, emb, out, world)
        try:
            got_h, got_l = sol.vocab_parallel_forward(
                tokens.copy(), emb.copy(), out.copy(), world
            )
        except Exception:
            return {"max_abs_err": float("inf")}

        got = np.concatenate(
            [np.asarray(got_h, dtype=np.float64).ravel(),
             np.asarray(got_l, dtype=np.float64).ravel()]
        )
        ref = np.concatenate([ref_h.ravel(), ref_l.ravel()])
        err = max(err, float(np.max(np.abs(got - ref))))
    return {"max_abs_err": err}
