def predict_ceiling_breaches(model_proto, limit_bytes=2147483648):
    breached = []
    for init in model_proto.graph.initializer:
        import numpy as np
        dtype_map = {
            1: np.float32, 2: np.uint8, 3: np.int8, 4: np.int16,
            5: np.int32, 6: np.int64, 7: np.string_, 9: np.bool_,
            10: np.float16, 11: np.double, 12: np.uint32, 13: np.uint64
        }
        np_type = dtype_map.get(init.data_type, np.float32)
        itemsize = np.dtype(np_type).itemsize
        num_elements = 1
        for dim in init.dims:
            num_elements *= dim
        total_bytes = num_elements * itemsize
        if total_bytes > limit_bytes:
            breached.append(init.name)
    return sorted(breached)
