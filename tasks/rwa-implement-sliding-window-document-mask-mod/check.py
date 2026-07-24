import numpy as np


def _oracle(Q, K, V, doc_ids, window):
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
    masked_logits = np.where(mask, logits, -np.inf)
    shifted = masked_logits - np.max(masked_logits, axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    out = probs @ V
    return out, mask


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([[1., 0.], [0., 1.], [1., 1.]]),
            np.array([[1., 0.], [0., 1.], [1., 1.]]),
            np.array([[1., 2.], [3., 4.], [5., 6.]]),
            np.array([0, 0, 1]),
            2,
        ),
        (
            np.arange(20, dtype=np.float64).reshape(5, 4) / 10.0,
            np.arange(20, dtype=np.float64).reshape(5, 4) / 7.0,
            np.arange(15, dtype=np.float64).reshape(5, 3),
            np.array([1, 1, 1, 2, 2]),
            3,
        ),
        (
            np.eye(6, dtype=np.float64),
            np.flip(np.eye(6, dtype=np.float64), axis=0),
            np.ones((6, 2), dtype=np.float64),
            np.array([0, 1, 0, 1, 0, 1]),
            4,
        ),
    ]

    max_err = 0.0
    mask_ok = 1.0
    for Q, K, V, docs, window in cases:
        ref_out, ref_mask = _oracle(Q, K, V, docs, window)
        try:
            got_out, got_mask = sol.sliding_window_document_attention(
                Q, K, V, docs, window
            )
            got_out = np.asarray(got_out, dtype=np.float64)
            got_mask = np.asarray(got_mask, dtype=bool)
        except Exception:
            return {"mask_match": 0.0, "max_abs_err": float("inf")}

        if got_mask.shape != ref_mask.shape or not np.array_equal(got_mask, ref_mask):
            mask_ok = 0.0

        if got_out.shape != ref_out.shape:
            max_err = float("inf")
        else:
            max_err = max(max_err, float(np.max(np.abs(got_out - ref_out))))

    return {
        "mask_match": mask_ok,
        "max_abs_err": max_err,
    }
