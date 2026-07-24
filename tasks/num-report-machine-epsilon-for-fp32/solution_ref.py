def machine_epsilon_fp32() -> int:
    import numpy as np
    return int(np.finfo(np.float32).eps.view(np.uint32))
