import numpy as np


def sliding_window_document_attention(Q, K, V, doc_ids, window):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    doc_ids = np.asarray(doc_ids)

    n, d = Q.shape
    q_idx = np.arange(n)[:, None]
    k_idx = np.arange(n)[None, :]
    mask = (
        (k_idx <= q_idx)
        & ((q_idx - k_idx) < window)
        & (doc_ids[:, None] == doc_ids[None, :])
    )

    logits = (Q @ K.T) / np.sqrt(d)
    logits = np.where(mask, logits, -np.inf)

    logits = logits - np.max(logits, axis=1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / np.sum(probs, axis=1, keepdims=True)

    return probs @ V, mask
