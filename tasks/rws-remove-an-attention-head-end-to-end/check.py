import numpy as np


def _softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def _oracle(Wq, Wk, Wv, Wo, x, head, num_heads):
    d = Wq.shape[1]
    head_dim = d // num_heads
    start = head * head_dim
    end = (head + 1) * head_dim

    Wq_p = np.concatenate([Wq[:, :start], Wq[:, end:]], axis=1)
    Wk_p = np.concatenate([Wk[:, :start], Wk[:, end:]], axis=1)
    Wv_p = np.concatenate([Wv[:, :start], Wv[:, end:]], axis=1)
    Wo_p = np.concatenate([Wo[:start, :], Wo[end:, :]], axis=0)

    q = x @ Wq_p
    k = x @ Wk_p
    v = x @ Wv_p

    kept_heads = num_heads - 1
    outputs = []
    for i in range(kept_heads):
        a = i * head_dim
        b = (i + 1) * head_dim
        scores = (q[:, a:b] @ k[:, a:b].T) / np.sqrt(head_dim)
        probs = _softmax(scores)
        outputs.append(probs @ v[:, a:b])

    concat = np.concatenate(outputs, axis=1)
    y = concat @ Wo_p
    return y


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    d = 12
    heads = 3
    n = 5
    Wq = rng.normal(size=(d, d))
    Wk = rng.normal(size=(d, d))
    Wv = rng.normal(size=(d, d))
    Wo = rng.normal(size=(d, d))
    x = rng.normal(size=(n, d))
    head = 1

    expected = _oracle(Wq, Wk, Wv, Wo, x, head, heads)

    try:
        result = sol.remove_attention_head(
            Wq, Wk, Wv, Wo, x, head, heads
        )
        got = np.asarray(result[-1], dtype=np.float64)
        err = float(np.max(np.abs(got - expected)))
    except Exception:
        err = float("inf")

    return {"max_abs_err": err}
