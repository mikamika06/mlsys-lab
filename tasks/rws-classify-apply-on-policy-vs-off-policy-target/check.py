import numpy as np

def _reference(tags, on_policy_targets, off_policy_targets):
    """
    Compute the expected routing mask and routed targets.
    """
    m = len(on_policy_targets) + len(off_policy_targets)
    if tags.shape[0] != m:
        raise ValueError("tags length does not match total number of targets")
    d = on_policy_targets.shape[1]
    routed = np.empty((m, d), dtype=np.float64)

    # Indices where tags are True (on‑policy) and False (off‑policy)
    on_idx = np.where(tags)[0]
    off_idx = np.where(~tags)[0]

    routed[on_idx] = on_policy_targets[:len(on_idx)]
    routed[off_idx] = off_policy_targets[:len(off_idx)]

    return tags.copy(), routed

def grade(sol, fx) -> dict:
    """
    Grade the candidate solution against a NumPy reference.
    """
    # Example test case – the grader will provide its own cases
    try:
        tags = np.array([True, False, True])
        on_policy_targets = np.array([[1., 2.], [3., 4.]])
        off_policy_targets = np.array([[5., 6.]])
        expected_mask, expected_routed = _reference(tags,
                                                    on_policy_targets,
                                                    off_policy_targets)
        got_mask, got_routed = sol.route_and_apply(tags,
                                                   on_policy_targets,
                                                   off_policy_targets)
    except Exception:
        return {"exact_match": 0.0}

    ok = np.array_equal(expected_mask, got_mask) and np.array_equal(expected_routed, got_routed)
    return {"exact_match": float(ok)}
