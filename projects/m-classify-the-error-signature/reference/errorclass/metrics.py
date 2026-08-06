import numpy as np


def compute_metrics(logits_ref, logits_target):
    ref = np.asarray(logits_ref, dtype=np.float32)
    tgt = np.asarray(logits_target, dtype=np.float32)

    top1_ref = np.argmax(ref, axis=-1)
    top1_tgt = np.argmax(tgt, axis=-1)

    agreement = float(np.mean(top1_ref == top1_tgt))

    diff = np.abs(ref - tgt)
    max_abs_diff = float(np.max(diff))
    mean_abs_diff = float(np.mean(diff))

    ref_norm = np.mean(np.abs(ref))
    mean_rel_diff = float(max_abs_diff / ref_norm) if ref_norm > 1e-6 else float(max_abs_diff)

    return {
        "top1_agreement": agreement,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "mean_rel_diff": mean_rel_diff,
        "has_nan": bool(np.isnan(tgt).any())
    }
