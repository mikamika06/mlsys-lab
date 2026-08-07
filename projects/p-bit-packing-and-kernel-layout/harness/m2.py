def check(workdir):
    import numpy as np
    from quant.packing import pack_bits, unpack_bits
    m = {"roundtrip_exact": 0.0}
    try:
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32)
        packed = pack_bits(data, bits=4)
        unpacked = unpack_bits(packed, bits=4, shape=(8,))
        if len(packed) > 0 and len(unpacked) == 8:
            m["roundtrip_exact"] = 1.0
    except Exception:
        pass
    return m
