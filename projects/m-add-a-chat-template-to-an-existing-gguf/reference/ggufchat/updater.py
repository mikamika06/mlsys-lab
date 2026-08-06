import struct
from ggufchat.template import validate_template


def _read_value(data: bytes, offset: int, val_type: int) -> tuple:
    if val_type == 0:
        return struct.unpack_from("<B", data, offset)[0], offset + 1
    if val_type == 1:
        return struct.unpack_from("<b", data, offset)[0], offset + 1
    if val_type == 2:
        return struct.unpack_from("<H", data, offset)[0], offset + 2
    if val_type == 3:
        return struct.unpack_from("<h", data, offset)[0], offset + 2
    if val_type == 4:
        return struct.unpack_from("<I", data, offset)[0], offset + 4
    if val_type == 5:
        return struct.unpack_from("<i", data, offset)[0], offset + 4
    if val_type == 6:
        return struct.unpack_from("<f", data, offset)[0], offset + 4
    if val_type == 7:
        return struct.unpack_from("<B", data, offset)[0] != 0, offset + 1
    if val_type == 8:
        slen = struct.unpack_from("<Q", data, offset)[0]
        sbytes = data[offset + 8 : offset + 8 + slen]
        return sbytes.decode("utf-8"), offset + 8 + slen
    if val_type == 9:
        elem_type, arr_len = struct.unpack_from("<IQ", data, offset)
        curr = offset + 12
        arr = []
        for _ in range(arr_len):
            item, curr = _read_value(data, curr, elem_type)
            arr.append(item)
        return (elem_type, arr), curr
    if val_type == 10:
        return struct.unpack_from("<Q", data, offset)[0], offset + 8
    if val_type == 11:
        return struct.unpack_from("<q", data, offset)[0], offset + 8
    if val_type == 12:
        return struct.unpack_from("<d", data, offset)[0], offset + 8
    raise ValueError(f"Unknown val_type: {val_type}")


def _pack_value(val, val_type: int) -> bytes:
    if val_type == 0:
        return struct.pack("<B", val)
    if val_type == 1:
        return struct.pack("<b", val)
    if val_type == 2:
        return struct.pack("<H", val)
    if val_type == 3:
        return struct.pack("<h", val)
    if val_type == 4:
        return struct.pack("<I", val)
    if val_type == 5:
        return struct.pack("<i", val)
    if val_type == 6:
        return struct.pack("<f", val)
    if val_type == 7:
        return struct.pack("<B", 1 if val else 0)
    if val_type == 8:
        sb = val.encode("utf-8")
        return struct.pack("<Q", len(sb)) + sb
    if val_type == 9:
        elem_type, items = val
        res = struct.pack("<IQ", elem_type, len(items))
        for item in items:
            res += _pack_value(item, elem_type)
        return res
    if val_type == 10:
        return struct.pack("<Q", val)
    if val_type == 11:
        return struct.pack("<q", val)
    if val_type == 12:
        return struct.pack("<d", val)
    raise ValueError(f"Unknown val_type: {val_type}")


def parse_gguf(data: bytes) -> tuple:
    """Parse GGUF file bytes into version, tensor_count, metadata, tensor_infos, alignment, payload."""
    if len(data) < 24 or data[:4] != b"GGUF":
        raise ValueError("Invalid GGUF magic or header")
    version, tensor_count, kv_count = struct.unpack_from("<IQQ", data, 4)
    offset = 20
    metadata = {}
    for _ in range(kv_count):
        klen = struct.unpack_from("<Q", data, offset)[0]
        offset += 8
        key = data[offset : offset + klen].decode("utf-8")
        offset += klen
        vtype = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        val, offset = _read_value(data, offset, vtype)
        metadata[key] = (vtype, val)

    alignment = 32
    if "general.alignment" in metadata:
        alignment = int(metadata["general.alignment"][1])

    tensors = []
    for _ in range(tensor_count):
        tname_len = struct.unpack_from("<Q", data, offset)[0]
        offset += 8
        tname = data[offset : offset + tname_len].decode("utf-8")
        offset += tname_len
        ndim = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        dims = [struct.unpack_from("<Q", data, offset + i * 8)[0] for i in range(ndim)]
        offset += ndim * 8
        ttype, toffset = struct.unpack_from("<IQ", data, offset)
        offset += 12
        tensors.append({"name": tname, "dims": dims, "type": ttype, "offset": toffset})

    raw_header_len = offset
    pad = (alignment - (raw_header_len % alignment)) % alignment
    payload_offset = raw_header_len + pad
    payload = data[payload_offset:]
    return version, tensor_count, metadata, tensors, alignment, payload


def add_or_update_chat_template(gguf_bytes: bytes, template_str: str) -> bytes:
    """Inject or update tokenizer.chat_template metadata in GGUF bytes."""
    if not validate_template(template_str):
        raise ValueError("Invalid chat template string")

    version, tensor_count, metadata, tensors, alignment, payload = parse_gguf(gguf_bytes)
    metadata["tokenizer.chat_template"] = (8, template_str)

    buf = b"GGUF" + struct.pack("<IQQ", version, tensor_count, len(metadata))
    for key, (vtype, val) in metadata.items():
        kb = key.encode("utf-8")
        buf += struct.pack("<Q", len(kb)) + kb + struct.pack("<I", vtype) + _pack_value(val, vtype)

    for t in tensors:
        tb = t["name"].encode("utf-8")
        buf += struct.pack("<Q", len(tb)) + tb + struct.pack("<I", len(t["dims"]))
        for d in t["dims"]:
            buf += struct.pack("<Q", d)
        buf += struct.pack("<IQ", t["type"], t["offset"])

    raw_header_len = len(buf)
    pad_len = (alignment - (raw_header_len % alignment)) % alignment
    buf += b"\x00" * pad_len
    return buf + payload
