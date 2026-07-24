import numpy as np


def _softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def remove_attention_head(Wq, Wk, Wv, Wo, x, head, num_heads):
    d = Wq.shape[1]
    head_dim = d // num_heads
    start = head * head_dim
    end = (head + 1) * head_dim

    Wq_p = np.concatenate([Wq[:, :start], Wq[:, end:]], axis=1)
    Wk_p = np.concatenate([Wk[:, :start], Wk[:, end:]], axis=1)
    Wv_p = np.concatenate([Wv[:, :start], Wv[:, end:]], axis=1)
    Wo_p = np.concatenate([Wo[:start], Wo[end:]], axis=0)

    q = x @ Wq_p
    k = x @ Wk_p
    v = x @ Wv_p

    outputs = []
    for i in range(num_heads - 1):
        a = i * head_dim
        b = (i + 1) * head_dim
        scores = (q[:, a:b] @ k[:, a:b].T) / np.sqrt(head_dim)
        outputs.append(_softmax(scores) @ v[:, a:b])

    y = np.concatenate(outputs, axis=1) @ Wo_p
    return Wq_p, Wk_p, Wv_p, Wo_p, y
