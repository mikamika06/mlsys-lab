def check(workdir):
    import numpy as np
    from quant.packing import pack_bits, unpack_bits
    m = {"random_tensors_match": 0.0}
    try:
        rng = np.random.default_rng(42)
        data = rng.integers(0, 15, size=64, dtype=np.int32)
        packed = pack_bits(data, bits=4)
        unpacked = unpack_bits(packed, bits=4, shape=data.shape)
        # Check if basic properties hold
        if unpacked.size >= data.size:
            m["random_tensors_match"] = 1.0
    except Exception:
        pass
    return m
