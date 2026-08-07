import struct

def classify_gguf(data):
    if len(data) < 4:
        return "TRUNCATED_HEADER"
    if data[:4] != b"GGUF":
        return "BAD_MAGIC"
    try:
        if len(data) < 16:
            return "TRUNCATED_HEADER"
        version = struct.unpack("<I", data[4:8])[0]
        n_kv = struct.unpack("<Q", data[8:16])[0]
        ptr = 16
        for _ in range(n_kv):
            if ptr + 8 > len(data):
                return "TRUNCATED_HEADER"
            k_len = struct.unpack("<Q", data[ptr:ptr+8])[0]
            ptr += 8 + k_len
            if ptr + 4 > len(data):
                return "TRUNCATED_HEADER"
            vtype = struct.unpack("<I", data[ptr:ptr+4])[0]
            ptr += 4
            if vtype not in (0, 1, 2, 3, 4, 5):
                return "INVALID_KV_TYPE"
            if vtype == 5:
                if ptr + 8 > len(data):
                    return "TRUNCATED_HEADER"
                v_len = struct.unpack("<Q", data[ptr:ptr+8])[0]
                ptr += 8 + v_len
            elif vtype in (2, 3, 4):
                ptr += 4
            else:
                ptr += 8
        if ptr > len(data):
            return "TRUNCATED_HEADER"
        if ptr % 4 != 0:
            return "BAD_ALIGNMENT"
        return "VALID"
    except Exception:
        return "TRUNCATED_HEADER"
