import numpy as np


def classify_signature(ref_tensor, target_tensor, metadata):
    ref = np.asarray(ref_tensor, dtype=np.float32)
    tgt = np.asarray(target_tensor, dtype=np.float32)

    if ref.shape != tgt.shape:
        return "SHAPE_MISMATCH"

    diff = np.abs(ref - tgt)
    max_abs_diff = float(np.max(diff))

    if np.isnan(tgt).any() or np.isinf(tgt).any():
        return "NUMERICAL_OVERFLOW"

    mean_ref = float(np.mean(np.abs(ref)))
    if mean_ref > 1e-6:
        rel_diff = max_abs_diff / mean_ref
    else:
        rel_diff = max_abs_diff

    if max_abs_diff < 1e-5:
        return "EXACT_MATCH"
    elif rel_diff < 1e-2:
        return "BENIGN_DRIFT"
    elif metadata.get("is_logits", False) and rel_diff > 0.5:
        return "CATASTROPHIC_DIVERGENCE"
    else:
        return "PRECISION_LOSS"
