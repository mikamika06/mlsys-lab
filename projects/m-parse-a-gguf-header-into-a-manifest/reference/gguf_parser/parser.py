import struct

TYPE_FORMATS = {
    0: ('<B', 1), 1: ('<b', 1), 2: ('<H', 2), 3: ('<h', 2),
    4: ('<I', 4), 5: ('<i', 4), 6: ('<f', 4), 7: ('?', 1),
    10: ('<Q', 8), 11: ('<q', 8), 12: ('<d', 8),
}

def decode_string(data: bytes, offset: int) -> tuple:
    length = struct.unpack_from('<Q', data, offset)[0]
    offset += 8
    val = data[offset:offset+length].decode('utf-8')
    return val, offset + length

def decode_value(data: bytes, offset: int, val_type: int) -> tuple:
    if val_type in TYPE_FORMATS:
        fmt, size = TYPE_FORMATS[val_type]
        val = struct.unpack_from(fmt, data, offset)[0]
        return val, offset + size
    elif val_type == 8:
        return decode_string(data, offset)
    elif val_type == 9:
        item_type = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        length = struct.unpack_from('<Q', data, offset)[0]
        offset += 8
        arr = []
        for _ in range(length):
            val, offset = decode_value(data, offset, item_type)
            arr.append(val)
        return arr, offset
    else:
        raise ValueError(f"Unknown type {val_type}")

def parse_header(data: bytes) -> dict:
    magic = data[0:4].decode('utf-8')
    version = struct.unpack_from('<I', data, 4)[0]
    tensor_count = struct.unpack_from('<Q', data, 8)[0]
    kv_count = struct.unpack_from('<Q', data, 16)[0]

    offset = 24
    metadata = {}
    for _ in range(kv_count):
        key, offset = decode_string(data, offset)
        val_type = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        val, offset = decode_value(data, offset, val_type)
        metadata[key] = val

    meta_end = offset

    tensors = []
    for _ in range(tensor_count):
        name, offset = decode_string(data, offset)
        n_dims = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        dims = []
        for _ in range(n_dims):
            dim = struct.unpack_from('<Q', data, offset)[0]
            offset += 8
            dims.append(dim)
        t_type = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        t_offset = struct.unpack_from('<Q', data, offset)[0]
        offset += 8
        tensors.append({
            "name": name,
            "n_dimensions": n_dims,
            "dimensions": dims,
            "type": t_type,
            "offset": t_offset
        })

    return {
        "magic": magic,
        "version": version,
        "tensor_count": tensor_count,
        "metadata_kv_count": kv_count,
        "metadata": metadata,
        "tensors": tensors,
        "header_end_offset": offset,
        "_meta_end": meta_end
    }

def compute_overhead(manifest: dict) -> dict:
    header_end = manifest["header_end_offset"]
    meta_end = manifest["_meta_end"]
    alignment = manifest["metadata"].get("general.alignment", 32)
    padding = (alignment - (header_end % alignment)) % alignment

    return {
        "metadata_bytes": meta_end - 24,
        "tensor_info_bytes": header_end - meta_end,
        "padding_waste": padding
    }
