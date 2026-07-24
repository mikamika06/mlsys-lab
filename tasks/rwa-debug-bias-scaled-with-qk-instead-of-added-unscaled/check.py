import numpy as np


def _oracle(q, k, v, bias, scale):
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)

    # Real production formula (matches torch.nn.functional.scaled_dot_product_attention
    # with a float attn_mask): logits = (q @ k^T) * scale, THEN add the
    # additive bias UNSCALED, then softmax.
    logits = (q @ k.T) * scale
    logits = logits + bias
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    w = np.exp(logits)
    w = w / np.sum(w, axis=-1, keepdims=True)
    return w @ v


def _cases():
    cases = []
    rng = np.random.default_rng(0)

    for _ in range(6):
        n_q = int(rng.integers(2, 6))
        n_k = int(rng.integers(3, 9))
        d = int(rng.integers(2, 8))
        dv = int(rng.integers(2, 8))

        q = rng.standard_normal((n_q, d))
        k = rng.standard_normal((n_k, d))
        v = rng.standard_normal((n_k, dv))
        bias = rng.standard_normal((n_q, n_k)) * 2.0
        scale = 1.0 / np.sqrt(d)
        cases.append((q, k, v, bias, scale))

    # ALiBi-style linear position bias, explicit non-default scale.
    n_q, n_k, d, dv = 4, 7, 5, 3
    q = rng.standard_normal((n_q, d))
    k = rng.standard_normal((n_k, d))
    v = rng.standard_normal((n_k, dv))
    positions = np.arange(n_k)[None, :] - np.arange(n_q)[:, None]
    bias = -0.3 * np.abs(positions).astype(np.float64)
    scale = 0.75 / np.sqrt(d)
    cases.append((q, k, v, bias, scale))

    # Zero bias edge case: correct and buggy formulas agree here, so it
    # alone cannot pass a broken implementation but confirms no regression
    # on the plain no-bias path.
    n_q, n_k, d, dv = 3, 5, 4, 4
    q = rng.standard_normal((n_q, d))
    k = rng.standard_normal((n_k, d))
    v = rng.standard_normal((n_k, dv))
    bias = np.zeros((n_q, n_k), dtype=np.float64)
    scale = 1.0 / np.sqrt(d)
    cases.append((q, k, v, bias, scale))

    return cases


def grade(sol, fx) -> dict:
    worst = 0.0
    for q, k, v, bias, scale in _cases():
        ref = _oracle(q, k, v, bias, scale)
        try:
            got = np.asarray(
                sol.sdpa_with_additive_bias(q, k, v, bias, scale), dtype=np.float64
            )
        except Exception:
            return {"max_abs_err": float("inf")}
        if got.shape != ref.shape:
            return {"max_abs_err": float("inf")}
        worst = max(worst, float(np.max(np.abs(got - ref))))
    return {"max_abs_err": worst}
