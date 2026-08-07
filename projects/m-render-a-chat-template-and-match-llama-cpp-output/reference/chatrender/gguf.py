import struct


def extract_chat_template(gguf_bytes: bytes) -> str:
    """Extract embedded chat template from GGUF bytes."""
    pos = 0
    if gguf_bytes[0:4] != b"GGUF":
        raise ValueError("Invalid GGUF magic")
    pos += 4
    _, _, kv_count = struct.unpack_from("<III", gguf_bytes, pos)
    pos += 12

    for _ in range(kv_count):
        key_len = struct.unpack_from("<Q", gguf_bytes, pos)[0]
        pos += 8
        key = gguf_bytes[pos:pos+key_len].decode("utf-8")
        pos += key_len

        value_type = struct.unpack_from("<I", gguf_bytes, pos)[0]
        pos += 4

        if value_type == 8:
            val_len = struct.unpack_from("<Q", gguf_bytes, pos)[0]
            pos += 8
            val = gguf_bytes[pos:pos+val_len].decode("utf-8")
            pos += val_len
            if key == "tokenizer.chat_template":
                return val
        elif value_type in (6, 7):
            pos += 4
        elif value_type in (4, 5):
            pos += 8
        elif value_type == 10:
            pos += 1
        elif value_type == 9:
            arr_type = struct.unpack_from("<I", gguf_bytes, pos)[0]
            pos += 4
            arr_len = struct.unpack_from("<Q", gguf_bytes, pos)[0]
            pos += 8
            if arr_type == 8:
                for _ in range(arr_len):
                    slen = struct.unpack_from("<Q", gguf_bytes, pos)[0]
                    pos += 8 + slen
            elif arr_type in (6, 7):
                pos += arr_len * 4
            elif arr_type in (4, 5):
                pos += arr_len * 8
            elif arr_type == 10:
                pos += arr_len * 1
        else:
            raise ValueError(f"Unknown GGUF value type {value_type}")
    raise KeyError("tokenizer.chat_template not found")
