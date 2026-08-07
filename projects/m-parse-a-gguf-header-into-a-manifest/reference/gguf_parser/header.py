import struct

GGUF_MAGIC = b"GGUF"

TYPE_UINT8 = 0
TYPE_INT8 = 1
TYPE_UINT16 = 2
TYPE_INT16 = 3
TYPE_UINT32 = 4
TYPE_INT32 = 5
TYPE_FLOAT32 = 6
TYPE_BOOL = 7
TYPE_STRING = 8
TYPE_ARRAY = 9
TYPE_UINT64 = 10
TYPE_INT64 = 11
TYPE_FLOAT64 = 12


def _read_string(data: bytes, offset: int) -> tuple[str, int]:
    length = struct.unpack_from("<Q", data, offset)[0]
    offset += 8
    val_bytes = data[offset : offset + length]
    offset += length
    return val_bytes.decode("utf-8"), offset


def _read_value(data: bytes, offset: int, vtype: int) -> tuple[object, int]:
    if vtype == TYPE_UINT8:
        return struct.unpack_from("<B", data, offset)[0], offset + 1
    elif vtype == TYPE_INT8:
        return struct.unpack_from("<b", data, offset)[0], offset + 1
    elif vtype == TYPE_UINT16:
        return struct.unpack_from("<H", data, offset)[0], offset + 2
    elif vtype == TYPE_INT16:
        return struct.unpack_from("<h", data, offset)[0], offset + 2
    elif vtype == TYPE_UINT32:
        return struct.unpack_from("<I", data, offset)[0], offset + 4
    elif vtype == TYPE_INT32:
        return struct.unpack_from("<i", data, offset)[0], offset + 4
    elif vtype == TYPE_FLOAT32:
        return struct.unpack_from("<f", data, offset)[0], offset + 4
    elif vtype == TYPE_BOOL:
        b = struct.unpack_from("<B", data, offset)[0]
        return bool(b), offset + 1
    elif vtype == TYPE_STRING:
        return _read_string(data, offset)
    elif vtype == TYPE_UINT64:
        return struct.unpack_from("<Q", data, offset)[0], offset + 8
    elif vtype == TYPE_INT64:
        return struct.unpack_from("<q", data, offset)[0], offset + 8
    elif vtype == TYPE_FLOAT64:
        return struct.unpack_from("<d", data, offset)[0], offset + 8
    elif vtype == TYPE_ARRAY:
        elem_type, count = struct.unpack_from("<IQ", data, offset)
        offset += 12
        arr = []
        for _ in range(count):
            elem, offset = _read_value(data, offset, elem_type)
            arr.append(elem)
        return arr, offset
    else:
        raise ValueError(f"Unknown value type: {vtype}")


def parse_gguf_header(data: bytes) -> dict:
    magic = data[:4]
    if magic != GGUF_MAGIC:
        raise ValueError("Invalid magic header")
    version, tensor_count, kv_count = struct.unpack_from("<III", data, 4)
    offset = 16
    kv_metadata = {}
    for _ in range(kv_count):
        key, offset = _read_string(data, offset)
        vtype = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        val, offset = _read_value(data, offset, vtype)
        kv_metadata[key] = val

    tensors = []
    for _ in range(tensor_count):
        name, offset = _read_string(data, offset)
        n_dims = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        dims = list(struct.unpack_from(f"<{n_dims}Q", data, offset))
        offset += 8 * n_dims
        type_code, tensor_offset = struct.unpack_from("<IQ", data, offset)
        offset += 12
        tensors.append(
            {
                "name": name,
                "n_dims": n_dims,
                "dimensions": dims,
                "type": type_code,
                "offset": tensor_offset,
            }
        )

    return {
        "version": version,
        "tensor_count": tensor_count,
        "kv_count": kv_count,
        "metadata": kv_metadata,
        "tensors": tensors,
        "header_size": offset,
    }
