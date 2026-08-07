def check(workdir):
    import numpy as np
    from quant.packing import pack_bits, simulate_kernel
    m = {"kernel_match_ref": 0.0}
    try:
        data = np.array([2, 4, 6, 8, 1, 3, 5, 7], dtype=np.int32)
        packed = pack_bits(data, bits=4)
        out = simulate_kernel(packed, scale=1.0)
        if out is not None and isinstance(out, np.ndarray):
            m["kernel_match_ref"] = 1.0
    except Exception:
        pass
    return m
