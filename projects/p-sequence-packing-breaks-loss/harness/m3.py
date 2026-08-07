import sys
import numpy as np
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    res = {
        "loss_ignores_padding": 0.0,
        "loss_normalization_ok": 0.0,
        "naive_vs_fixed_differs": 0.0,
    }

    try:
        from seqpack.loss import compute_naive_packed_loss, compute_packed_loss
    except Exception:
        return res

    rng = np.random.RandomState(42)
    L, V = 10, 20
    logits = rng.randn(L, V)
    labels = np.array([1, 2, 3, 4, 5, 0, 0, 0, 0, 0], dtype=np.int64)
    label_mask = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    seq_ids = np.array([0, 0, 0, 1, 1, -1, -1, -1, -1, -1], dtype=np.int64)

    try:
        l1 = compute_packed_loss(logits, labels, label_mask, seq_ids)

        logits_pad_changed = logits.copy()
        logits_pad_changed[5:] = rng.randn(5, V)
        labels_pad_changed = labels.copy()
        labels_pad_changed[5:] = 19

        l2 = compute_packed_loss(logits_pad_changed, labels_pad_changed, label_mask, seq_ids)

        if abs(l1 - l2) < 1e-6 and l1 > 0:
            res["loss_ignores_padding"] = 1.0
    except Exception:
        pass

    try:
        exp_lg = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_lg / np.sum(exp_lg, axis=-1, keepdims=True)
        valid_idx = np.where((label_mask > 0) & (seq_ids >= 0))[0]
        expected_loss = np.mean(-np.log(probs[valid_idx, labels[valid_idx]]))

        l_calc = compute_packed_loss(logits, labels, label_mask, seq_ids)
        if abs(l_calc - expected_loss) < 1e-5:
            res["loss_normalization_ok"] = 1.0
    except Exception:
        pass

    try:
        naive_loss = compute_naive_packed_loss(logits, labels, label_mask)
        fixed_loss = compute_packed_loss(logits, labels, label_mask, seq_ids)
        if abs(naive_loss - fixed_loss) > 1e-3:
            res["naive_vs_fixed_differs"] = 1.0
    except Exception:
        pass

    return res
