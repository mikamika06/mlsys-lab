import numpy as np

def route_and_apply(tags: np.ndarray,
                    on_policy_targets: np.ndarray,
                    off_policy_targets: np.ndarray):
    """
    Correct implementation of routing tokens to the appropriate target set.
    """
    tags = np.asarray(tags, dtype=bool)
    on_policy_targets = np.asarray(on_policy_targets, dtype=np.float64)
    off_policy_targets = np.asarray(off_policy_targets, dtype=np.float64)

    m = len(on_policy_targets) + len(off_policy_targets)
    if tags.shape[0] != m:
        raise ValueError("tags length must equal total number of targets")

    d = on_policy_targets.shape[1]
    routed = np.empty((m, d), dtype=np.float64)

    on_count = 0
    off_count = 0
    for i in range(m):
        if tags[i]:
            for j in range(d):
                routed[i, j] = on_policy_targets[on_count, j]
            on_count += 1
        else:
            for j in range(d):
                routed[i, j] = off_policy_targets[off_count, j]
            off_count += 1

    return tags.copy(), routed
