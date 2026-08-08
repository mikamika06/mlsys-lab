import struct

GGUF_MAGIC = 0x46554747
VALUE_TYPE_STRING = 8

def _read_string(data: bytes, offset: int) -> tuple[str, int]:
    length = struct.unpack_from("<Q", data, offset)[0]
    offset += 8
    s = data[offset:offset + length].decode("utf-8")
    return s, offset + length

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

    data_base = offset + ((alignment - (offset % alignment)) % alignment)

    out = bytearray(gguf_bytes)
    for k, v in patches.items():
        if k not in kv_positions:
            continue
        val_off, orig_len = kv_positions[k]
        encoded = v.encode("utf-8")
        if len(encoded) != orig_len:
            raise ValueError(f"Patch string length mismatch for {k}: expected {orig_len}, got {len(encoded)}")
        out[val_off + 8:val_off + 8 + orig_len] = encoded

    if bytes(out[data_base:]) != bytes(gguf_bytes[data_base:]):
        raise RuntimeError("Tensor bytes corrupted during metadata patching")

    return bytes(out)
