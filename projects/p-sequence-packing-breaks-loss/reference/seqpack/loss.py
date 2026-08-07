import numpy as np


def compute_naive_packed_loss(logits, labels, label_mask):
    """Computes naive packed loss ignoring sequence boundary constraints."""
    lg = np.asarray(logits, dtype=np.float64)
    lbl = np.asarray(labels, dtype=np.int64)

    L, V = lg.shape
    exp_lg = np.exp(lg - np.max(lg, axis=-1, keepdims=True))
    probs = exp_lg / np.sum(exp_lg, axis=-1, keepdims=True)

    safe_lbl = np.clip(lbl, 0, V - 1)
    loss_per_token = -np.log(probs[np.arange(L), safe_lbl] + 1e-12)

    return float(np.mean(loss_per_token))


def compute_packed_loss(logits, labels, label_mask, seq_ids):
    """Computes token cross-entropy loss correctly normalized across valid targets."""
    lg = np.asarray(logits, dtype=np.float64)
    lbl = np.asarray(labels, dtype=np.int64)
    msk = np.asarray(label_mask, dtype=np.float64)
    s_ids = np.asarray(seq_ids, dtype=np.int64)

    L, V = lg.shape
    valid_mask = (msk > 0) & (s_ids >= 0) & (lbl >= 0) & (lbl < V)

    if not np.any(valid_mask):
        return 0.0

    exp_lg = np.exp(lg - np.max(lg, axis=-1, keepdims=True))
    probs = exp_lg / np.sum(exp_lg, axis=-1, keepdims=True)

    safe_lbl = np.clip(lbl, 0, V - 1)
    loss_per_token = -np.log(probs[np.arange(L), safe_lbl] + 1e-12)

    valid_loss = loss_per_token * valid_mask
    total_valid_tokens = np.sum(valid_mask)

    return float(np.sum(valid_loss) / total_valid_tokens)
