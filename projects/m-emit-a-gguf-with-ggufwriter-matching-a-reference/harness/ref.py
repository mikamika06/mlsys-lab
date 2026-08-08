import struct

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3

VALUE_TYPE_UINT32 = 4
VALUE_TYPE_FLOAT32 = 6
VALUE_TYPE_STRING = 8

def _encode_string(s: str) -> bytes:
    encoded = s.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded

def _read_string(data: bytes, offset: int) -> tuple[str, int]:
    length = struct.unpack_from("<Q", data, offset)[0]
    offset += 8
    s = data[offset:offset + length].decode("utf-8")
    return s, offset + length

class GGUFWriter:
    def __init__(self, alignment: int = 32) -> None:
        self.alignment = alignment
        self.kv = []
        self.tensors = []

    def add_uint32(self, key: str, val: int) -> None:
        self.kv.append((key, VALUE_TYPE_UINT32, struct.pack("<I", val)))

    def add_float32(self, key: str, val: float) -> None:
        self.kv.append((key, VALUE_TYPE_FLOAT32, struct.pack("<f", val)))

    def add_string(self, key: str, val: str) -> None:
        self.kv.append((key, VALUE_TYPE_STRING, _encode_string(val)))

    def add_tensor(self, name: str, shape: list[int], dtype_id: int, data: bytes) -> None:
        self.tensors.append((name, shape, dtype_id, data))

    def write(self) -> bytes:
        header = struct.pack("<IIQQ", GGUF_MAGIC, GGUF_VERSION, len(self.tensors), len(self.kv) + 1)
        kv_bytes = bytearray(header)
        kv_bytes.extend(_encode_string("general.alignment"))
        kv_bytes.extend(struct.pack("<I", VALUE_TYPE_UINT32))
        kv_bytes.extend(struct.pack("<I", self.alignment))

        for k, vtype, val_bytes in self.kv:
            kv_bytes.extend(_encode_string(k))
            kv_bytes.extend(struct.pack("<I", vtype))
            kv_bytes.extend(val_bytes)

        t_headers = bytearray()
        offset = 0
        padded_tensors = []

        for name, shape, dtype_id, data in self.tensors:
            t_headers.extend(_encode_string(name))
            t_headers.extend(struct.pack("<I", len(shape)))
            for dim in shape:
                t_headers.extend(struct.pack("<Q", dim))
            t_headers.extend(struct.pack("<I", dtype_id))
            t_headers.extend(struct.pack("<Q", offset))

            pad_data = (self.alignment - (len(data) % self.alignment)) % self.alignment
            padded_data = data + b"\x00" * pad_data
            padded_tensors.append(padded_data)
            offset += len(padded_data)

        meta_bytes = kv_bytes + t_headers
        pad_meta = (self.alignment - (len(meta_bytes) % self.alignment)) % self.alignment
        result = bytearray(meta_bytes) + b"\x00" * pad_meta

        for pt in padded_tensors:
            result.extend(pt)

        return bytes(result)

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

def patch_metadata_in_place(gguf_bytes: bytes, patches: dict[str, str]) -> bytes:
    magic, version, tensor_count, kv_count = struct.unpack_from("<IIQQ", gguf_bytes, 0)
    if magic != GGUF_MAGIC:
        raise ValueError("Invalid GGUF magic")

    offset = 24
    alignment = 32
    kv_positions = {}

    for _ in range(kv_count):
        key, next_off = _read_string(gguf_bytes, offset)
        vtype = struct.unpack_from("<I", gguf_bytes, next_off)[0]
        val_off = next_off + 4
        if vtype == 4:
            val = struct.unpack_from("<I", gguf_bytes, val_off)[0]
            if key == "general.alignment":
                alignment = val
            offset = val_off + 4
        elif vtype == 6:
            offset = val_off + 4
        elif vtype == VALUE_TYPE_STRING:
            length = struct.unpack_from("<Q", gguf_bytes, val_off)[0]
            kv_positions[key] = (val_off, length)
            offset = val_off + 8 + length

    for _ in range(tensor_count):
        _, offset = _read_string(gguf_bytes, offset)
        n_dims = struct.unpack_from("<I", gguf_bytes, offset)[0]
        offset += 4 + (n_dims * 8) + 12

    out = bytearray(gguf_bytes)
    for k, v in patches.items():
        if k not in kv_positions:
            continue
        val_off, orig_len = kv_positions[k]
        encoded = v.encode("utf-8")
        if len(encoded) != orig_len:
            raise ValueError(f"Patch string length mismatch for {k}: expected {orig_len}, got {len(encoded)}")
        out[val_off + 8:val_off + 8 + orig_len] = encoded

    return bytes(out)

TEST_SPECS = [
    {
        "alignment": 32,
        "uints": [("llama.block_count", 12), ("llama.context_length", 2048)],
        "floats": [("llama.rope.dimension_scale", 1.0)],
        "strings": [("general.architecture", "llama"), ("general.name", "test-llama-7b")],
        "tensors": [
            ("token_embd.weight", [32, 16], 0, bytes([i % 256 for i in range(512)])),
            ("output.weight", [32, 16], 0, bytes([(i * 5) % 256 for i in range(512)])),
        ],
    },
    {
        "alignment": 64,
        "uints": [("mpt.block_count", 24)],
        "floats": [("mpt.attn_clamp", 8.0)],
        "strings": [("general.architecture", "mpt-model"), ("general.author", "community")],
        "tensors": [
            ("attn.qkv.weight", [64, 32], 1, bytes([(i + 7) % 256 for i in range(2048)])),
        ],
    },
]
