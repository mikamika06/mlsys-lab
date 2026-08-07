import numpy as np


def generate_cases():
    np.random.seed(123)
    cases = []
    for _ in range(5):
        b = int(np.random.randint(1, 3))
        h = int(np.random.randint(2, 5))
        n_q = int(np.random.randint(16, 33))
        n_k = int(np.random.randint(16, 33))
        d = int(np.random.choice([16, 32, 64]))
        q = np.random.randn(b, h, n_q, d).astype(np.float32)
        k = np.random.randn(b, h, n_k, d).astype(np.float32)
        v = np.random.randn(b, h, n_k, d).astype(np.float32)
        cases.append((q, k, v))
    return cases


def oracle_sdpa_to_flash(q, k, v):
    return (
        np.transpose(q, (0, 2, 1, 3)),
        np.transpose(k, (0, 2, 1, 3)),
        np.transpose(v, (0, 2, 1, 3)),
    )


def oracle_reference_attention(q, k, v):
    b, h, n_q, d = q.shape
    scale = 1.0 / np.sqrt(d)
    scores = np.matmul(q, np.transpose(k, (0, 1, 3, 2))) * scale
    max_scores = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    sum_exp = np.sum(exp_scores, axis=-1, keepdims=True)
    attn_weights = exp_scores / sum_exp
    out = np.matmul(attn_weights, v)
    lse = max_scores.squeeze(-1) + np.log(sum_exp.squeeze(-1))
    return out, lse
