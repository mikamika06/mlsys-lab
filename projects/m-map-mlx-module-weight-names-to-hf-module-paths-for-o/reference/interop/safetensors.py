import json
import struct
import numpy as np


def compute_safetensors_header(tensors: dict) -> tuple:
    dtype_map = {
        np.dtype("float32"): "F32",
        np.dtype("float16"): "F16",
        np.dtype("float64"): "F64",
        np.dtype("int32"): "I32",
        np.dtype("int64"): "I64",
        np.dtype("int16"): "I16",
        np.dtype("int8"): "I8",
        np.dtype("uint8"): "U8",
        np.dtype("bool"): "BOOL",
    }
    header = {}
    offset = 0
    for name in sorted(tensors.keys()):
        arr = tensors[name]
        size = arr.nbytes
        dt = dtype_map.get(arr.dtype, "F32")
        header[name] = {
            "data_offsets": [offset, offset + size],
            "dtype": dt,
            "shape": list(arr.shape),
        }
        offset += size

    header_json = json.dumps(header, separators=(",", ":"), sort_keys=True)
    header_bytes = header_json.encode("utf-8")
    header_len = len(header_bytes)
    prefix_bytes = struct.pack("<Q", header_len)
    return header_len, prefix_bytes
