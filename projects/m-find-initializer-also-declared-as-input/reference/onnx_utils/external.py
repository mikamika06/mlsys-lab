def resolve_external_ranges(model_proto):
    resolved = {}
    for init in model_proto.graph.initializer:
        ext_location = None
        ext_offset = 0
        ext_length = 0
        for entry in init.external_data:
            if entry.key == "location":
                ext_location = entry.value
            elif entry.key == "offset":
                ext_offset = int(entry.value) if entry.value else 0
            elif entry.key == "length":
                ext_length = int(entry.value) if entry.value else 0
        if ext_location:
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
            size_bytes = num_elements * itemsize
            length = ext_length if ext_length > 0 else size_bytes
            resolved[init.name] = {
                "location": ext_location,
                "offset": ext_offset,
                "length": length,
                "start": ext_offset,
                "end": ext_offset + length
            }
    return resolved
