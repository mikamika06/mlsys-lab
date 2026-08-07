import os
import sys
import numpy as np
import ref


def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    try:
        from repack.permute import get_marlin_perm_map, permute_weights, unpermute_weights
    except Exception as e:
        return {"byte_exact_fraction": 0.0, "_note": f"Import error: {e}"}

    total = float(len(ref.CONFIGS))
    passed = 0.0

    for K, N in ref.CONFIGS:
        W = ref.generate_test_matrix(K, N)

        want_perm_map = ref.get_marlin_perm_map(K, N)
        try:
            got_perm_map = get_marlin_perm_map(K, N)
            if not np.array_equal(got_perm_map, want_perm_map):
                return {"byte_exact_fraction": passed / total, "_note": f"mismatch in perm map for ({K}, {N})"}
        except Exception as e:
            return {"byte_exact_fraction": passed / total, "_note": f"get_marlin_perm_map failed: {e}"}

        want_perm_W = ref.permute_weights(W)
        try:
            got_perm_W = permute_weights(W)
            if not np.array_equal(got_perm_W, want_perm_W):
                return {"byte_exact_fraction": passed / total, "_note": f"mismatch in permuted weights for ({K}, {N})"}
        except Exception as e:
            return {"byte_exact_fraction": passed / total, "_note": f"permute_weights failed: {e}"}

        try:
            got_unperm = unpermute_weights(got_perm_W, K, N)
            if not np.array_equal(got_unperm, W):
                return {"byte_exact_fraction": passed / total, "_note": f"unpermute_weights failed to restore original matrix for ({K}, {N})"}
        except Exception as e:
            return {"byte_exact_fraction": passed / total, "_note": f"unpermute_weights failed: {e}"}

        passed += 1.0

    return {"byte_exact_fraction": passed / total}
