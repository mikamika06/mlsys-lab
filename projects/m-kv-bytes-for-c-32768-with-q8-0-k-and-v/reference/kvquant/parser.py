import struct


def parse_gguf_kv_params(data: bytes) -> dict:
    if len(data) < 12:
        raise ValueError("Buffer too short for GGUF magic and version")
    magic = data[:4]
    if magic != b"GGUF":
        raise ValueError("Invalid GGUF magic bytes")
    version, n_tensors, n_kv = struct.unpack("<III", data[4:16])
    idx = 16

    def _read_str(offset):
        length = struct.unpack("<Q", data[offset : offset + 8])[0]
        offset += 8
        s = data[offset : offset + length].decode("utf-8")
        return s, offset + length

    def _read_value(vtype, offset):
        if vtype == 4:
            val = struct.unpack("<I", data[offset : offset + 4])[0]
            return val, offset + 4
        elif vtype == 5:
            val = struct.unpack("<i", data[offset : offset + 4])[0]
            return val, offset + 4
        elif vtype == 10:
            val = struct.unpack("<Q", data[offset : offset + 8])[0]
            return val, offset + 8
        elif vtype == 11:
            val = struct.unpack("<q", data[offset : offset + 8])[0]
            return val, offset + 8
        elif vtype == 8:
            return _read_str(offset)
        else:
            raise ValueError(f"Unsupported value type {vtype}")

    target_keys = {
        "block_count",
        "feed_forward_length",
        "embedding_length",
        "head_count",
        "head_count_kv",
        "context_length",
    }
    res = {}

    for _ in range(n_kv):
        key, idx = _read_str(idx)
        vtype = struct.unpack("<I", data[idx : idx + 4])[0]
        idx += 4
        val, idx = _read_value(vtype, idx)

        for tk in target_keys:
            if key.endswith("." + tk) or key == tk:
                res[tk] = int(val)

    return res
