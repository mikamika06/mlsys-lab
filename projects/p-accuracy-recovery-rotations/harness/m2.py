def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    import ref
    import numpy as np
    from quant_rec.rotations import apply_rotation

    m = {"rotation_applied": 0.0, "outliers_reduced": 0.0}
    weights, _, matrix, _ = ref.get_test_data()
    rotated, max_val = apply_rotation(weights, matrix)
    if isinstance(rotated, np.ndarray) and rotated.shape == weights.shape:
        m["rotation_applied"] = 1.0
    if isinstance(max_val, float):
        m["outliers_reduced"] = 1.0
    return m
