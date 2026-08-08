import struct

GGUF_MAGIC = 0x46554747

VALUE_TYPE_UINT32 = 4
VALUE_TYPE_FLOAT32 = 6
VALUE_TYPE_STRING = 8

def _read_string(data: bytes, offset: int) -> tuple[str, int]:
    length = struct.unpack_from("<Q", data, offset)[0]
    offset += 8
    s = data[offset:offset + length].decode("utf-8")
    return s, offset + length

def dump_json(gguf_bytes: bytes) -> dict:
    magic, version, tensor_count, kv_count = struct.unpack_from("<IIQQ", gguf_bytes, 0)
    if magic != GGUF_MAGIC:
        raise ValueError("Invalid GGUF magic")

    offset = 24
    metadata = {}
    alignment = 32

    for _ in range(kv_count):
        key, offset = _read_string(gguf_bytes, offset)
        vtype = struct.unpack_from("<I", gguf_bytes, offset)[0]
        offset += 4
        if vtype == VALUE_TYPE_UINT32:
            val = struct.unpack_from("<I", gguf_bytes, offset)[0]
            offset += 4
            metadata[key] = val
            if key == "general.alignment":
                alignment = val
        elif vtype == VALUE_TYPE_FLOAT32:
            val = struct.unpack_from("<f", gguf_bytes, offset)[0]
            offset += 4
            metadata[key] = round(val, 6)
        elif vtype == VALUE_TYPE_STRING:
            val, offset = _read_string(gguf_bytes, offset)
            metadata[key] = val

    tensors = []
    for _ in range(tensor_count):
        name, offset = _read_string(gguf_bytes, offset)
        n_dims = struct.unpack_from("<I", gguf_bytes, offset)[0]
        offset += 4
        shape = []
        for _ in range(n_dims):
            dim = struct.unpack_from("<Q", gguf_bytes, offset)[0]
            offset += 8
            shape.append(dim)
        dtype_id, t_offset = struct.unpack_from("<IQ", gguf_bytes, offset)
        offset += 12
        tensors.append({"name": name, "shape": shape, "dtype": dtype_id, "offset": t_offset})

    data_base = offset + ((alignment - (offset % alignment)) % alignment)

    for t in tensors:
        start = data_base + t["offset"]
        size = 1
        for d in t["shape"]:
            size *= d
        t["size_bytes"] = size
        t["data_preview"] = gguf_bytes[start:start + min(4, size)].hex()

    return {
        "version": version,
        "alignment": alignment,
        "metadata": metadata,
        "tensors": tensors,
    }
