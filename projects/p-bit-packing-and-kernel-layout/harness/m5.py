def check(workdir):
    import numpy as np
    from quant.packing import pack_bits, simulate_kernel
    m = {"functional_ok": 0.0}
    try:
        data = np.zeros(32, dtype=np.int32)
        packed = pack_bits(data, bits=4)
        out = simulate_kernel(packed, scale=2.0)
        if np.all(out == 0.0):
            m["functional_ok"] = 1.0
    except Exception:
        pass
    return m
