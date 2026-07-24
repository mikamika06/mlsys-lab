import numpy as np


def _oracle(x, q_proj, k_proj, v_proj, o_proj, heads, keep_heads):
    n, d = x.shape
    head_dim = d // heads

    q = x @ q_proj
    k = x @ k_proj
    v = x @ v_proj

    q = q.reshape(n, heads, head_dim)
    k = k.reshape(n, heads, head_dim)
    v = v.reshape(n, heads, head_dim)

    outputs = []
    for h in keep_heads:
        scores = q[:, h, :] @ k[:, h, :].T / np.sqrt(head_dim)
        scores = scores - np.max(scores, axis=1, keepdims=True)
        probs = np.exp(scores)
        probs = probs / np.sum(probs, axis=1, keepdims=True)
        outputs.append(probs @ v[:, h, :])

    z = np.concatenate(outputs, axis=1)
    cols = []
    for h in keep_heads:
        cols.extend(range(h * head_dim, (h + 1) * head_dim))
    return z @ o_proj[cols, :]


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    cases = []
    for heads, d, n, keep in [
        (2, 4, 3, [0]),
        (4, 8, 5, [1, 3]),
        (4, 12, 2, [0, 2, 3]),
    ]:
        x = rng.normal(size=(n, d))
        q = rng.normal(size=(d, d))
        k = rng.normal(size=(d, d))
        v = rng.normal(size=(d, d))
        o = rng.normal(size=(d, d))
        cases.append((x, q, k, v, o, heads, keep))

    max_err = 0.0
    for case in cases:
        try:
            got = sol.pruned_attention_forward(*case)
        except Exception:
            return {"max_abs_err": float("inf")}
        ref = _oracle(*case)
        err = float(np.max(np.abs(np.asarray(got) - ref)))
        max_err = max(max_err, err)

    return {"max_abs_err": max_err}
