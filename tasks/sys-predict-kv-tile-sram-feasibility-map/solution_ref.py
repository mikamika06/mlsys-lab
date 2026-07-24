import numpy as np


def kv_tile_sram_feasibility_map(configs):
    arr = np.asarray(configs, dtype=np.int64)
    required = arr[:, 0] * arr[:, 1] * 4
    return list(required <= arr[:, 2])
