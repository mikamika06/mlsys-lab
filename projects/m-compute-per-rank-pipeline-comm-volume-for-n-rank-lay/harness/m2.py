import os
import sys
import numpy as np
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    try:
        from mlxdist.sharding import derive_load_balanced_sharding
        from mlxdist.ring import launch_2rank_ring_all_reduce
    except ImportError:
        return {"sharding_balanced": 0.0, "ring_reduced": 0.0, "_note": "Import error"}

    weights = [1.2, 2.5, 1.8, 3.1, 0.9, 2.2, 1.4, 2.8]
    want_sharding = ref.derive_load_balanced_sharding(len(weights), weights, 4)

    try:
        got_sharding = derive_load_balanced_sharding(len(weights), weights, 4)
    except Exception as e:
        return {"sharding_balanced": 0.0, "ring_reduced": 0.0, "_note": f"Sharding error: {e}"}

    sharding_ok = (got_sharding == want_sharding)

    t_a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    t_b = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], dtype=np.float32)

    try:
        got_ring = launch_2rank_ring_all_reduce(t_a, t_b)
        ref_ring = ref.launch_2rank_ring_all_reduce(t_a, t_b)

        r0_ok = np.allclose(got_ring["rank0"], ref_ring["rank0"])
        r1_ok = np.allclose(got_ring["rank1"], ref_ring["rank1"])
        ring_ok = r0_ok and r1_ok
    except Exception as e:
        return {"sharding_balanced": 1.0 if sharding_ok else 0.0, "ring_reduced": 0.0, "_note": f"Ring error: {e}"}

    return {
        "sharding_balanced": 1.0 if sharding_ok else 0.0,
        "ring_reduced": 1.0 if ring_ok else 0.0,
    }
