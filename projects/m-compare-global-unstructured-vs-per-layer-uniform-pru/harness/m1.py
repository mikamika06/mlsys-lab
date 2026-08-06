"""Checker for Milestone 1: Global and Uniform Mask Generation."""

import numpy as np
import ref


def check(workdir):
    """Check mask and threshold precision relative to reference."""
    try:
        from prunecomp.pruners import (
            compute_global_unstructured_mask,
            compute_per_layer_uniform_masks,
        )
    except Exception as e:
        return {"rel_err": 1.0, "_note": f"Import error: {e}"}

    weights = ref.generate_test_model(seed=999)
    ratio = 0.35

    try:
        got_u_masks, got_u_thresh = compute_per_layer_uniform_masks(weights, ratio)
        got_g_masks, got_g_thresh = compute_global_unstructured_mask(
            weights, ratio
        )

        ref_u_masks, ref_u_thresh = ref.reference_uniform_prune(weights, ratio)
        ref_g_masks, ref_g_thresh = ref.reference_global_prune(weights, ratio)
    except Exception as e:
        return {"rel_err": 1.0, "_note": f"Execution error: {e}"}

    errors = []

    for name in weights:
        u_diff = np.mean(got_u_masks[name] != ref_u_masks[name])
        g_diff = np.mean(got_g_masks[name] != ref_g_masks[name])
        errors.extend([u_diff, g_diff])

        t_ref = ref_u_thresh[name]
        t_got = got_u_thresh[name]
        err_t = abs(t_got - t_ref) / (abs(t_ref) + 1e-12)
        errors.append(err_t)

    err_g = abs(got_g_thresh - ref_g_thresh) / (abs(ref_g_thresh) + 1e-12)
    errors.append(err_g)

    max_err = float(np.max(errors))
    return {"rel_err": max_err}
