import json
import struct
import numpy as np

DTYPE_MAP_REVERSE = {
    "F16": np.float16,
    "F32": np.float32,
    "I32": np.int32,
    "I64": np.int64,
}


def parse_safetensors_bytes(st_bytes: bytes) -> dict:
    if len(st_bytes) < 8:
        raise ValueError("File too short for safetensors header")
    header_len = struct.unpack("<Q", st_bytes[:8])[0]
    if len(st_bytes) < 8 + header_len:
        raise ValueError("File truncated before header end")

    header_json = st_bytes[8 : 8 + header_len].decode("utf-8")
    header = json.loads(header_json)
    data_start = 8 + header_len

    tensors = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        dtype_str = meta["dtype"]
        shape = meta["shape"]
        start, end = meta["data_offsets"]
        raw_data = st_bytes[data_start + start : data_start + end]

        np_dtype = DTYPE_MAP_REVERSE.get(dtype_str, np.float32)
        arr = np.frombuffer(raw_data, dtype=np_dtype).reshape(shape)

        tensors[name] = {
            "dtype": dtype_str,
            "shape": shape,
            "data": raw_data,
            "array": arr,
        }
    return tensors
