import struct


def update_context_length_inplace(file_bytes: bytearray, new_ctx_len: int) -> bytearray:
    """Rewrite context_length in-place within GGUF header without altering tensor offset."""
    if file_bytes[:4] != b"GGUF":
        raise ValueError("Invalid GGUF magic header")

    version = struct.unpack_from("<I", file_bytes, 4)[0]
    if version not in (2, 3):
        raise ValueError(f"Unsupported GGUF version: {version}")

    tensor_count, kv_count = struct.unpack_from("<QQ", file_bytes, 8)

    offset = 24
    for _ in range(kv_count):
        key_len = struct.unpack_from("<Q", file_bytes, offset)[0]
        offset += 8
        key = file_bytes[offset:offset + key_len].decode("utf-8")
        offset += key_len

        val_type = struct.unpack_from("<I", file_bytes, offset)[0]
        offset += 4

        if key in ("llm.context_length", "llama.context_length", "general.context_length"):
            if val_type in (4, 5):
                struct.pack_into("<I", file_bytes, offset, new_ctx_len)
            elif val_type in (10, 11):
                struct.pack_into("<Q", file_bytes, offset, new_ctx_len)
            return file_bytes

        if val_type in (0, 1, 2, 3, 6, 7, 8):
            offset += 1
        elif val_type in (4, 5, 9):
            offset += 4
        elif val_type in (10, 11):
            offset += 8
        elif val_type == 12:
            offset += 16
        elif val_type == 8:
            str_len = struct.unpack_from("<Q", file_bytes, offset)[0]
            offset += 8 + str_len
        elif val_type == 9:
            arr_type = struct.unpack_from("<I", file_bytes, offset)[0]
            arr_len = struct.unpack_from("<Q", file_bytes, offset + 4)[0]
            offset += 12
            if arr_type == 8:
                for _ in range(arr_len):
                    s_len = struct.unpack_from("<Q", file_bytes, offset)[0]
                    offset += 8 + s_len
            elif arr_type in (4, 5):
                offset += 4 * arr_len
            elif arr_type in (10, 11):
                offset += 8 * arr_len
            else:
                offset += arr_len
        else:
            raise ValueError(f"Unhandled metadata type {val_type}")

    raise KeyError("Context length key not found in metadata header")
