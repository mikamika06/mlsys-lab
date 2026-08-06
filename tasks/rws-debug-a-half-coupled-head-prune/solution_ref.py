import numpy as np


def pruned_attention_forward(x, q_proj, k_proj, v_proj, o_proj, heads, keep_heads):
    n, d = x.shape
    head_dim = d // heads

    q = (x @ q_proj).reshape(n, heads, head_dim)
    k = (x @ k_proj).reshape(n, heads, head_dim)
    v = (x @ v_proj).reshape(n, heads, head_dim)

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
