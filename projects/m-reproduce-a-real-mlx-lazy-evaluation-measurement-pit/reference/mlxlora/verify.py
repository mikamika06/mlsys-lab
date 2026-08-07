import numpy as np

def verify_base_weights_untouched(initial_weights, trained_weights, target_modules=None):
    missing_keys = []
    modified_keys = []
    checked_count = 0

    for k, v_init in initial_weights.items():
        if k not in trained_weights:
            missing_keys.append(k)
            continue

        v_curr = trained_weights[k]
        checked_count += 1

        if v_init.shape != v_curr.shape or v_init.dtype != v_curr.dtype:
            modified_keys.append(k)
            continue

        if not np.array_equal(v_init, v_curr):
            modified_keys.append(k)

    all_frozen_matched = (len(missing_keys) == 0) and (len(modified_keys) == 0)

    return {
        "all_frozen_matched": all_frozen_matched,
        "modified_keys": sorted(modified_keys),
        "missing_keys": sorted(missing_keys),
        "checked_count": checked_count
    }
