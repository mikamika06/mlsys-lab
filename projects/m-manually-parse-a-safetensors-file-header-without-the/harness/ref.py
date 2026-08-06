import json
import re
import struct
import numpy as np

DTYPE_MAP = {
    np.dtype("float32"): "F32",
    np.dtype("float16"): "F16",
    np.dtype("int32"): "I32",
    np.dtype("int64"): "I64",
}

DTYPE_MAP_REVERSE = {
    "F16": np.float16,
    "F32": np.float32,
    "I32": np.int32,
    "I64": np.int64,
}

np.random.seed(42)

TENSORS_1 = {
    "model.layers.0.self_attn.q_proj.weight": np.random.randn(4, 8).astype(
        np.float16
    ),
    "model.layers.0.self_attn.k_proj.weight": np.random.randn(4, 8).astype(
        np.float16
    ),
    "model.embed_tokens.weight": np.random.randn(16, 8).astype(np.float16),
}

TENSORS_2 = {
    "blk.0.attn_q.weight": np.random.randn(8, 16).astype(np.float16),
    "output.weight": np.random.randn(16, 32).astype(np.float32),
}

HF_TO_MLX_RULES = {
    "model.layers.{i}.self_attn.q_proj.weight": "layers.{i}.attention.wq.weight",
    "model.layers.{i}.self_attn.k_proj.weight": "layers.{i}.attention.wk.weight",
    "model.embed_tokens.weight": "embed_tokens.weight",
}

UNTRANSLATED_HF_WEIGHTS = {
    "model.layers.0.self_attn.q_proj.weight": np.ones((4, 4), dtype=np.float16),
    "model.layers.0.self_attn.k_proj.weight": np.ones((4, 4), dtype=np.float16),
    "model.embed_tokens.weight": np.ones((8, 4), dtype=np.float16),
    "model.unknown_layer.weight": np.ones((2, 2), dtype=np.float16),
}


def make_safetensors(tensor_map):
    header = {}
    payload = bytearray()
    curr_offset = 0

    for name, arr in tensor_map.items():
        data_bytes = arr.tobytes()
        length = len(data_bytes)
        dtype_str = DTYPE_MAP.get(arr.dtype, "F32")
        header[name] = {
            "dtype": dtype_str,
            "shape": list(arr.shape),
            "data_offsets": [curr_offset, curr_offset + length],
        }
        payload.extend(data_bytes)
        curr_offset += length

    header_json = json.dumps(header).encode("utf-8")
    header_len = len(header_json)
    out = bytearray()
    out.extend(struct.pack("<Q", header_len))
    out.extend(header_json)
    out.extend(payload)
    return bytes(out)


def make_gguf(tensor_map, alignment=32):
    out = bytearray()
    out.extend(b"GGUF")
    out.extend(struct.pack("<I", 3))
    out.extend(struct.pack("<Q", len(tensor_map)))
    out.extend(struct.pack("<Q", 0))

    data_payload = bytearray()
    tensor_infos = []
    current_data_offset = 0

    for name, arr in tensor_map.items():
        name_bytes = name.encode("utf-8")
        dtype_code = 1 if arr.dtype == np.float16 else 0
        arr_bytes = arr.tobytes()

        rem = current_data_offset % alignment
        if rem != 0:
            pad = alignment - rem
            data_payload.extend(b"\x00" * pad)
            current_data_offset += pad

        tensor_infos.append(
            {
                "name_bytes": name_bytes,
                "shape": list(arr.shape),
                "type": dtype_code,
                "offset": current_data_offset,
            }
        )
        data_payload.extend(arr_bytes)
        current_data_offset += len(arr_bytes)

    for info in tensor_infos:
        out.extend(struct.pack("<Q", len(info["name_bytes"])))
        out.extend(info["name_bytes"])
        out.extend(struct.pack("<I", len(info["shape"])))
        for dim in reversed(info["shape"]):
            out.extend(struct.pack("<Q", dim))
        out.extend(struct.pack("<I", info["type"]))
        out.extend(struct.pack("<Q", info["offset"]))

    header_len = len(out)
    rem = header_len % alignment
    if rem != 0:
        out.extend(b"\x00" * (alignment - rem))

    out.extend(data_payload)
    return bytes(out)


def parse_safetensors_reference(st_bytes: bytes) -> dict:
    if len(st_bytes) < 8:
        raise ValueError("File too short for safetensors header")
    header_len = struct.unpack("<Q", st_bytes[:8])[0]
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


def parse_gguf_reference(gguf_bytes: bytes, alignment: int = 32) -> dict:
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


def verify_f16_bit_identity_reference(
    st_bytes: bytes, gguf_bytes: bytes
) -> dict:
    st_tensors = parse_safetensors_reference(st_bytes)
    gguf_tensors = parse_gguf_reference(gguf_bytes)

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


def remap_hf_to_mlx_reference(
    hf_tensors: dict, rule_map: dict
) -> tuple[dict, list[str]]:
    remapped = {}
    unmapped = []
    for key, val in hf_tensors.items():
        new_key = None
        for pattern, target_template in rule_map.items():
            if "{i}" in pattern:
                regex_pattern = (
                    "^"
                    + re.escape(pattern).replace(r"\{i\}", r"(?P<i>\d+)")
                    + "$"
                )
                match = re.match(regex_pattern, key)
                if match:
                    new_key = target_template.format(**match.groupdict())
                    break
            elif pattern == key:
                new_key = target_template
                break

        if new_key is not None:
            remapped[new_key] = val
        else:
            unmapped.append(key)
    return remapped, unmapped
