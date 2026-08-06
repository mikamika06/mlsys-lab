import struct
import numpy as np

GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_I32 = 2
GGML_TYPE_I16 = 3
GGML_TYPE_I8 = 4


def _read_string(buf, offset):
    length = struct.unpack_from(">Q", buf, offset)[0]
    offset += 8
    s = bytes(buf[offset : offset + length]).decode("utf-8")
    offset += length
    return s, offset


def _read_val(buf, offset, val_type):
    if val_type == GGUF_TYPE_UINT8:
        return struct.unpack_from(">B", buf, offset)[0], offset + 1
    if val_type == GGUF_TYPE_INT8:
        return struct.unpack_from(">b", buf, offset)[0], offset + 1
    if val_type == GGUF_TYPE_UINT16:
        return struct.unpack_from(">H", buf, offset)[0], offset + 2
    if val_type == GGUF_TYPE_INT16:
        return struct.unpack_from(">h", buf, offset)[0], offset + 2
    if val_type == GGUF_TYPE_UINT32:
        return struct.unpack_from(">I", buf, offset)[0], offset + 4
    if val_type == GGUF_TYPE_INT32:
        return struct.unpack_from(">i", buf, offset)[0], offset + 4
    if val_type == GGUF_TYPE_UINT64:
        return struct.unpack_from(">Q", buf, offset)[0], offset + 8
    if val_type == GGUF_TYPE_INT64:
        return struct.unpack_from(">q", buf, offset)[0], offset + 8
    if val_type == GGUF_TYPE_FLOAT32:
        return struct.unpack_from(">f", buf, offset)[0], offset + 4
    if val_type == GGUF_TYPE_FLOAT64:
        return struct.unpack_from(">d", buf, offset)[0], offset + 8
    if val_type == GGUF_TYPE_BOOL:
        return bool(struct.unpack_from(">?", buf, offset)[0]), offset + 1
    if val_type == GGUF_TYPE_STRING:
        return _read_string(buf, offset)
    if val_type == GGUF_TYPE_ARRAY:
        elem_type = struct.unpack_from(">I", buf, offset)[0]
        offset += 4
        count = struct.unpack_from(">Q", buf, offset)[0]
        offset += 8
        items = []
        for _ in range(count):
            elem, offset = _read_val(buf, offset, elem_type)
            items.append(elem)
        return items, offset
    raise ValueError(f"Unknown GGUF val type: {val_type}")


def read_gguf_be(buf, default_alignment=32):
    """Read big-endian GGUF binary buffer and return metadata_kv, tensor_infos, data_base_offset."""
    if not isinstance(buf, (bytes, memoryview, bytearray)):
        buf = memoryview(buf)

    offset = 0
    magic = bytes(buf[offset : offset + 4])
    if magic != b"GGUF":
        raise ValueError(f"Invalid GGUF magic: {magic}")
    offset += 4

    version = struct.unpack_from(">I", buf, offset)[0]
    if version != 3:
        raise ValueError(f"Unsupported GGUF version: {version}")
    offset += 4

    tensor_count = struct.unpack_from(">Q", buf, offset)[0]
    offset += 8

    kv_count = struct.unpack_from(">Q", buf, offset)[0]
    offset += 8

    metadata_kv = {}
    for _ in range(kv_count):
        key, offset = _read_string(buf, offset)
        vtype = struct.unpack_from(">I", buf, offset)[0]
        offset += 4
        val, offset = _read_val(buf, offset, vtype)
        metadata_kv[key] = val

    tensor_infos = []
    for _ in range(tensor_count):
        name, offset = _read_string(buf, offset)
        n_dims = struct.unpack_from(">I", buf, offset)[0]
        offset += 4
        shape = []
        for _ in range(n_dims):
            dim = struct.unpack_from(">Q", buf, offset)[0]
            offset += 8
            shape.append(dim)
        ggml_type = struct.unpack_from(">I", buf, offset)[0]
        offset += 4
        t_offset = struct.unpack_from(">Q", buf, offset)[0]
        offset += 8
        tensor_infos.append(
            {
                "name": name,
                "shape": tuple(shape),
                "ggml_type": ggml_type,
                "offset": t_offset,
            }
        )

    alignment = metadata_kv.get("general.alignment", default_alignment)
    pad = (alignment - (offset % alignment)) % alignment
    data_base_offset = offset + pad

    return metadata_kv, tensor_infos, data_base_offset
