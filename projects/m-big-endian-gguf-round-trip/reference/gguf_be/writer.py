import io
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

DTYPE_TO_GGML = {
    "float32": GGML_TYPE_F32,
    "float16": GGML_TYPE_F16,
    "int32": GGML_TYPE_I32,
    "int16": GGML_TYPE_I16,
    "int8": GGML_TYPE_I8,
}

GGML_TO_BE_DTYPE = {
    GGML_TYPE_F32: np.dtype(">f4"),
    GGML_TYPE_F16: np.dtype(">f2"),
    GGML_TYPE_I32: np.dtype(">i4"),
    GGML_TYPE_I16: np.dtype(">i2"),
    GGML_TYPE_I8: np.dtype("i1"),
}


def _encode_string(s):
    b = s.encode("utf-8")
    return struct.pack(">Q", len(b)) + b


def _encode_val(val, val_type):
    if val_type == GGUF_TYPE_UINT8:
        return struct.pack(">B", val)
    if val_type == GGUF_TYPE_INT8:
        return struct.pack(">b", val)
    if val_type == GGUF_TYPE_UINT16:
        return struct.pack(">H", val)
    if val_type == GGUF_TYPE_INT16:
        return struct.pack(">h", val)
    if val_type == GGUF_TYPE_UINT32:
        return struct.pack(">I", val)
    if val_type == GGUF_TYPE_INT32:
        return struct.pack(">i", val)
    if val_type == GGUF_TYPE_UINT64:
        return struct.pack(">Q", val)
    if val_type == GGUF_TYPE_INT64:
        return struct.pack(">q", val)
    if val_type == GGUF_TYPE_FLOAT32:
        return struct.pack(">f", float(val))
    if val_type == GGUF_TYPE_FLOAT64:
        return struct.pack(">d", float(val))
    if val_type == GGUF_TYPE_BOOL:
        return struct.pack(">?", bool(val))
    if val_type == GGUF_TYPE_STRING:
        return _encode_string(str(val))
    if val_type == GGUF_TYPE_ARRAY:
        elem_type, items = val
        buf = struct.pack(">I", elem_type) + struct.pack(">Q", len(items))
        for item in items:
            buf += _encode_val(item, elem_type)
        return buf
    raise ValueError(f"Unsupported GGUF val type: {val_type}")


def _infer_type(val):
    if isinstance(val, bool):
        return GGUF_TYPE_BOOL, val
    if isinstance(val, int):
        if val >= 0:
            return (
                (GGUF_TYPE_UINT32, val)
                if val <= 0xFFFFFFFF
                else (GGUF_TYPE_UINT64, val)
            )
        return (
            (GGUF_TYPE_INT32, val)
            if val >= -0x80000000
            else (GGUF_TYPE_INT64, val)
        )
    if isinstance(val, float):
        return GGUF_TYPE_FLOAT32, val
    if isinstance(val, str):
        return GGUF_TYPE_STRING, val
    if isinstance(val, list):
        if not val:
            return GGUF_TYPE_ARRAY, (GGUF_TYPE_INT32, [])
        first_t, _ = _infer_type(val[0])
        return GGUF_TYPE_ARRAY, (first_t, val)
    if isinstance(val, tuple) and len(val) == 2:
        return GGUF_TYPE_ARRAY, val
    raise ValueError(f"Cannot infer type for {type(val)}")


def write_gguf_be(metadata_kv, tensors, alignment=32):
    """Write a big-endian GGUF container into bytes."""
    stream = io.BytesIO()
    stream.write(b"GGUF")
    stream.write(struct.pack(">I", 3))
    stream.write(struct.pack(">Q", len(tensors)))
    stream.write(struct.pack(">Q", len(metadata_kv)))

    for k, v in metadata_kv.items():
        stream.write(_encode_string(k))
        if isinstance(v, tuple) and len(v) == 2 and isinstance(v[0], int):
            vtype, vval = v
        else:
            vtype, vval = _infer_type(v)
        stream.write(struct.pack(">I", vtype))
        stream.write(_encode_val(vval, vtype))

    tensor_entries = []
    current_relative_offset = 0

    for t in tensors:
        name = t["name"]
        data = np.asarray(t["data"])
        dtype_str = str(t.get("dtype", str(data.dtype)))
        ggml_type = t.get("ggml_type", DTYPE_TO_GGML.get(dtype_str, GGML_TYPE_F32))
        be_dtype = GGML_TO_BE_DTYPE[ggml_type]

        if current_relative_offset % alignment != 0:
            current_relative_offset += (
                alignment - (current_relative_offset % alignment)
            )

        tensor_entries.append(
            {
                "name": name,
                "shape": tuple(data.shape),
                "ggml_type": ggml_type,
                "offset": current_relative_offset,
                "data_be": data.astype(be_dtype).tobytes(),
            }
        )
        current_relative_offset += len(tensor_entries[-1]["data_be"])

    for entry in tensor_entries:
        stream.write(_encode_string(entry["name"]))
        shape = entry["shape"]
        stream.write(struct.pack(">I", len(shape)))
        for dim in shape:
            stream.write(struct.pack(">Q", dim))
        stream.write(struct.pack(">I", entry["ggml_type"]))
        stream.write(struct.pack(">Q", entry["offset"]))

    curr_pos = stream.tell()
    pad = (alignment - (curr_pos % alignment)) % alignment
    stream.write(b"\x00" * pad)

    base_offset = stream.tell()

    for entry in tensor_entries:
        curr = stream.tell() - base_offset
        if curr < entry["offset"]:
            stream.write(b"\x00" * (entry["offset"] - curr))
        stream.write(entry["data_be"])

    return stream.getvalue()
