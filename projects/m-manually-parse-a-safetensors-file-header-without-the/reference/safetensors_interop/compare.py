import struct
import numpy as np
from safetensors_interop.header import parse_safetensors_bytes


def parse_gguf_bytes(gguf_bytes: bytes, alignment: int = 32) -> dict:
    if gguf_bytes[:4] != b"GGUF":
        raise ValueError("Invalid GGUF magic header")
    version = struct.unpack("<I", gguf_bytes[4:8])[0]
    tensor_count, metadata_kv_count = struct.unpack("<QQ", gguf_bytes[8:24])

    pos = 24
    for _ in range(metadata_kv_count):
        klen = struct.unpack("<Q", gguf_bytes[pos : pos + 8])[0]
        pos += 8 + klen
        vtype = struct.unpack("<I", gguf_bytes[pos : pos + 4])[0]
        pos += 4
        if vtype in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11):
            sizes = {
                0: 1,
                1: 1,
                2: 2,
                3: 2,
                4: 4,
                5: 4,
                6: 4,
                7: 8,
                8: 8,
                9: 8,
                10: 4,
                11: 8,
            }
            pos += sizes[vtype]
        elif vtype == 12:
            slen = struct.unpack("<Q", gguf_bytes[pos : pos + 8])[0]
            pos += 8 + slen
        elif vtype == 13:
            arr_type, arr_len = struct.unpack("<IQ", gguf_bytes[pos : pos + 12])
            pos += 12

    tensor_infos = []
    for _ in range(tensor_count):
        nlen = struct.unpack("<Q", gguf_bytes[pos : pos + 8])[0]
        pos += 8
        name = gguf_bytes[pos : pos + nlen].decode("utf-8")
        pos += nlen
        n_dims = struct.unpack("<I", gguf_bytes[pos : pos + 4])[0]
        pos += 4
        dims = []
        for _ in range(n_dims):
            dims.append(struct.unpack("<Q", gguf_bytes[pos : pos + 8])[0])
            pos += 8
        shape = list(reversed(dims))
        type_code = struct.unpack("<I", gguf_bytes[pos : pos + 4])[0]
        pos += 4
        offset = struct.unpack("<Q", gguf_bytes[pos : pos + 8])[0]
        pos += 8

        tensor_infos.append(
            {
                "name": name,
                "shape": shape,
                "type": type_code,
                "offset": offset,
            }
        )

    rem = pos % alignment
    body_start = pos if rem == 0 else pos + (alignment - rem)

    tensors = {}
    for info in tensor_infos:
        num_elements = 1
        for d in info["shape"]:
            num_elements *= d
        elem_size = 2 if info["type"] == 1 else 4
        byte_len = num_elements * elem_size
        start = body_start + info["offset"]
        raw = gguf_bytes[start : start + byte_len]

        dtype_str = "F16" if info["type"] == 1 else "F32"
        np_dtype = np.float16 if info["type"] == 1 else np.float32
        arr = np.frombuffer(raw, dtype=np_dtype).reshape(info["shape"])

        tensors[info["name"]] = {
            "dtype": dtype_str,
            "shape": info["shape"],
            "data": raw,
            "array": arr,
        }
    return tensors


def verify_f16_bit_identity(st_bytes: bytes, gguf_bytes: bytes) -> dict:
    st_tensors = parse_safetensors_bytes(st_bytes)
    gguf_tensors = parse_gguf_bytes(gguf_bytes)

    matched = []
    mismatched = []

    for name, st_info in st_tensors.items():
        if name not in gguf_tensors:
            continue
        gguf_info = gguf_tensors[name]

        if st_info["dtype"] != "F16" or gguf_info["dtype"] != "F16":
            continue

        st_u16 = np.frombuffer(st_info["data"], dtype=np.uint16)
        gguf_u16 = np.frombuffer(gguf_info["data"], dtype=np.uint16)

        if st_info["shape"] == gguf_info["shape"] and np.array_equal(
            st_u16, gguf_u16
        ):
            matched.append(name)
        else:
            mismatched.append(name)

    bit_identical = (len(mismatched) == 0) and (len(matched) > 0)
    return {
        "matched_tensors": matched,
        "mismatched_tensors": mismatched,
        "bit_identical": bit_identical,
    }
