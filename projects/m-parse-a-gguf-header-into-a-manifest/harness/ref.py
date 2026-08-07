import struct

def parse_header(data: bytes) -> dict:
    TYPE_FORMATS = {
        0: ('<B', 1), 1: ('<b', 1), 2: ('<H', 2), 3: ('<h', 2),
        4: ('<I', 4), 5: ('<i', 4), 6: ('<f', 4), 7: ('?', 1),
        10: ('<Q', 8), 11: ('<q', 8), 12: ('<d', 8),
    }

    def decode_string(offset: int) -> tuple:
        length = struct.unpack_from('<Q', data, offset)[0]
        offset += 8
        val = data[offset:offset+length].decode('utf-8')
        return val, offset + length

    def decode_value(offset: int, val_type: int) -> tuple:
        if val_type in TYPE_FORMATS:
            fmt, size = TYPE_FORMATS[val_type]
            val = struct.unpack_from(fmt, data, offset)[0]
            return val, offset + size
        elif val_type == 8:
            return decode_string(offset)
        elif val_type == 9:
            item_type = struct.unpack_from('<I', data, offset)[0]
            offset += 4
            length = struct.unpack_from('<Q', data, offset)[0]
            offset += 8
            arr = []
            for _ in range(length):
                val, offset = decode_value(offset, item_type)
                arr.append(val)
            return arr, offset
        else:
            raise ValueError(f"Unknown type {val_type}")

    magic = data[0:4].decode('utf-8')
    version = struct.unpack_from('<I', data, 4)[0]
    tensor_count = struct.unpack_from('<Q', data, 8)[0]
    kv_count = struct.unpack_from('<Q', data, 16)[0]

    offset = 24
    metadata = {}
    for _ in range(kv_count):
        key, offset = decode_string(offset)
        val_type = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        val, offset = decode_value(offset, val_type)
        metadata[key] = val

    meta_end = offset

    tensors = []
    for _ in range(tensor_count):
        name, offset = decode_string(offset)
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

def build_gguf(metadata: dict, tensors: list) -> bytes:
    out = b"GGUF"
    out += struct.pack("<I", 3)
    out += struct.pack("<Q", len(tensors))
    out += struct.pack("<Q", len(metadata))

    def pack_string(s: str) -> bytes:
        b = s.encode('utf-8')
        return struct.pack("<Q", len(b)) + b

    def pack_val(val, val_type) -> bytes:
        TYPE_FORMATS = {
            0: ('<B', 1), 1: ('<b', 1), 2: ('<H', 2), 3: ('<h', 2),
            4: ('<I', 4), 5: ('<i', 4), 6: ('<f', 4), 7: ('?', 1),
            10: ('<Q', 8), 11: ('<q', 8), 12: ('<d', 8),
        }
        if val_type in TYPE_FORMATS:
            return struct.pack(TYPE_FORMATS[val_type][0], val)
        elif val_type == 8:
            return pack_string(val)
        elif val_type == 9:
            item_type, items = val
            res = struct.pack("<I", item_type)
            res += struct.pack("<Q", len(items))
            for item in items:
                res += pack_val(item, item_type)
            return res
        raise ValueError()

    for k, (v_type, v) in metadata.items():
        out += pack_string(k)
        out += struct.pack("<I", v_type)
        out += pack_val(v, v_type)

    for t in tensors:
        out += pack_string(t["name"])
        out += struct.pack("<I", t["n_dimensions"])
        for dim in t["dimensions"]:
            out += struct.pack("<Q", dim)
        out += struct.pack("<I", t["type"])
        out += struct.pack("<Q", t["offset"])

    return out

FIXTURES = [
    build_gguf(
        {
            "general.alignment": (4, 64),
            "name": (8, "test_model"),
            "nested_array": (9, (9, [(4, [1, 2]), (4, [3, 4])]))
        },
        [{"name": "tensor1", "n_dimensions": 2, "dimensions": [128, 64], "type": 1, "offset": 0}]
    ),
    build_gguf(
        {
            "t0": (0, 255), "t1": (1, -128), "t2": (2, 65535), "t3": (3, -32768),
            "t4": (4, 4294967295), "t5": (5, -2147483648), "t6": (6, 3.14159),
            "t7": (7, True), "t8": (8, "str"),
            "t10": (10, 18446744073709551615), "t11": (11, -9223372036854775808),
            "t12": (12, 2.718281828)
        },
        [{"name": "tensorA", "n_dimensions": 1, "dimensions": [1024], "type": 0, "offset": 0},
         {"name": "tensorB", "n_dimensions": 1, "dimensions": [1024], "type": 0, "offset": 1024}]
    ),
    build_gguf(
        {},
        []
    )
]
