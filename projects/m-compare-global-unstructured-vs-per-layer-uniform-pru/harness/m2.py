"""Checker for Milestone 2: Retention and Sparsity Metrics."""

import numpy as np
import ref


def check(workdir):
    """Check correctness of metric computations."""
    out = {"metrics_match": 0.0}
    try:
        from prunecomp.metrics import evaluate_pruning_quality
        from prunecomp.pruners import (
            compute_global_unstructured_mask,
            compute_per_layer_uniform_masks,
        )
    except Exception as e:
        out["_note"] = f"Import error: {e}"
        return out

    weights = ref.generate_test_model(seed=777)
    ratio = 0.45

    ref_u_masks, _ = ref.reference_uniform_prune(weights, ratio)
    ref_g_masks, _ = ref.reference_global_prune(weights, ratio)

    try:
        u_metrics = evaluate_pruning_quality(weights, ref_u_masks)
        g_metrics = evaluate_pruning_quality(weights, ref_g_masks)
    except Exception as e:
        out["_note"] = f"Evaluation error: {e}"
        return out

    for name, w in weights.items():
        elems = w.size
        m = ref_u_masks[name]
        pruned_cnt = int(np.sum(~m))
        expected_sp = float(pruned_cnt) / float(elems)

        got_sp = u_metrics["per_layer"][name]["sparsity"]
        if not np.isclose(got_sp, expected_sp, atol=1e-6):
            out["_note"] = f"Layer {name} uniform sparsity mismatch"
            return out

        w_ret = np.where(m, w, 0.0)
        expected_frob = np.linalg.norm(w_ret) / np.linalg.norm(w)
        got_frob = u_metrics["per_layer"][name]["frobenius_retention"]
        if not np.isclose(got_frob, expected_frob, atol=1e-6):
            out["_note"] = f"Layer {name} frobenius retention mismatch"
            return out

    g_sparsities = [g_metrics["per_layer"][k]["sparsity"] for k in weights]
    if np.allclose(g_sparsities[0], g_sparsities[1]):
        out["_note"] = (
            "Global pruning should yield non-uniform layer sparsities for heterogeneous layers"
        )
        return out

    out["metrics_match"] = 1.0
    return out
