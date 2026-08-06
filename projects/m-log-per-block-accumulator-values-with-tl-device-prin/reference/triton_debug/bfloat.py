def check_bfloat16_behavior(val):
    import numpy as np
    arr = np.asarray([val], dtype=np.float32)
    truncated = arr.astype(np.float32)
    return float(truncated[0])
