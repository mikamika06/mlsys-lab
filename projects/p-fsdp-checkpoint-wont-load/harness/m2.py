import numpy as np
import ref

def check(workdir):
    m = {"sharding_mapped": 0.0}
    try:
        import fsdp_ckpt.converter as conv
        state = {"weight": np.zeros((4, 2), dtype=np.float32)}
        mapped = conv.map_sharding(state, 2)
        if isinstance(mapped, dict) and "weight" in mapped:
            m["sharding_mapped"] = 1.0
    except Exception:
        pass
    return m
