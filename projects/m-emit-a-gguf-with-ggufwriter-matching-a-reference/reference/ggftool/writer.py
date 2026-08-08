import struct

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3

VALUE_TYPE_UINT32 = 4
VALUE_TYPE_FLOAT32 = 6
VALUE_TYPE_STRING = 8

class GGUFWriter:
    def __init__(self, alignment: int = 32) -> None:
        self.alignment = alignment
        self.kv = []
        self.tensors = []

    def _encode_string(self, s: str) -> bytes:
        encoded = s.encode("utf-8")
        return struct.pack("<Q", len(encoded)) + encoded

    def add_uint32(self, key: str, val: int) -> None:
        self.kv.append((key, VALUE_TYPE_UINT32, struct.pack("<I", val)))

    def add_float32(self, key: str, val: float) -> None:
        self.kv.append((key, VALUE_TYPE_FLOAT32, struct.pack("<f", val)))

    def add_string(self, key: str, val: str) -> None:
        self.kv.append((key, VALUE_TYPE_STRING, self._encode_string(val)))

    def add_tensor(self, name: str, shape: list[int], dtype_id: int, data: bytes) -> None:
        self.tensors.append((name, shape, dtype_id, data))

    def write(self) -> bytes:
        header = struct.pack("<IIQQ", GGUF_MAGIC, GGUF_VERSION, len(self.tensors), len(self.kv) + 1)
        kv_bytes = bytearray(header)
        kv_bytes.extend(self._encode_string("general.alignment"))
        kv_bytes.extend(struct.pack("<I", VALUE_TYPE_UINT32))
        kv_bytes.extend(struct.pack("<I", self.alignment))

        for k, vtype, val_bytes in self.kv:
            kv_bytes.extend(self._encode_string(k))
            kv_bytes.extend(struct.pack("<I", vtype))
            kv_bytes.extend(val_bytes)

        t_headers = bytearray()
        offset = 0
        padded_tensors = []

        for name, shape, dtype_id, data in self.tensors:
            t_headers.extend(self._encode_string(name))
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
