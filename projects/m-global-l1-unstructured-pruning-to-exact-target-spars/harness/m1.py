import ref
import numpy as np

def check(workdir):
    from prune.global_prune import global_magnitude_prune
    weights = ref.get_test_weights()
    target = 0.8
    got = global_magnitude_prune(weights, target)
    want = ref.global_magnitude_prune(weights, target)

    match = 1.0
    for name in weights:
        if name not in got or not np.allclose(got[name], want[name]):
            match = 0.0
            break
    return {"sparsity_match": match}
