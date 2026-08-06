"""Evaluation and metric comparisons for pruned model layers."""

import numpy as np


def evaluate_pruning_quality(weights, masks):
    """Compute quality metrics comparing original weights and pruned masks."""
    per_layer = {}
    total_elements = 0
    total_pruned = 0
    total_retained_sq = 0.0
    total_orig_sq = 0.0

    for name, w in weights.items():
        m = masks[name]
        elems = w.size
        pruned_cnt = int(np.sum(~m))
        sparsity = float(pruned_cnt) / float(elems) if elems > 0 else 0.0

        orig_frob_sq = float(np.sum(w**2))
        retained_w = np.where(m, w, 0.0)
        retained_frob_sq = float(np.sum(retained_w**2))

        frob_retention = (
            np.sqrt(retained_frob_sq) / np.sqrt(orig_frob_sq)
            if orig_frob_sq > 0
            else 1.0
        )

        per_layer[name] = {
            "sparsity": float(sparsity),
            "frobenius_retention": float(frob_retention),
            "elements": elems,
            "pruned": pruned_cnt,
        }

        total_elements += elems
        total_pruned += pruned_cnt
        total_retained_sq += retained_frob_sq
        total_orig_sq += orig_frob_sq

    overall_sparsity = (
        float(total_pruned) / float(total_elements) if total_elements > 0 else 0.0
    )
    overall_frob_retention = (
        np.sqrt(total_retained_sq) / np.sqrt(total_orig_sq)
        if total_orig_sq > 0
        else 1.0
    )

    return {
        "per_layer": per_layer,
        "overall_sparsity": overall_sparsity,
        "overall_frobenius_retention": overall_frob_retention,
    }
