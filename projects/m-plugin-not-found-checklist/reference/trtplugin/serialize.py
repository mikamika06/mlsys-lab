import struct

TYPE_FLOAT = 1
TYPE_INT = 2
TYPE_INT_ARRAY = 3
TYPE_STRING = 4


def serialize_fields(fields_dict):
    buffer = bytearray()
    buffer.extend(struct.pack("<I", len(fields_dict)))
    for key in sorted(fields_dict.keys()):
        val = fields_dict[key]
        key_bytes = key.encode("utf-8")
        buffer.extend(struct.pack("<H", len(key_bytes)))
        buffer.extend(key_bytes)

        if isinstance(val, float):
            buffer.extend(struct.pack("<B", TYPE_FLOAT))
            buffer.extend(struct.pack("<f", val))
        elif isinstance(val, int):
            buffer.extend(struct.pack("<B", TYPE_INT))
            buffer.extend(struct.pack("<q", val))
        elif isinstance(val, list) and all(isinstance(x, int) for x in val):
            buffer.extend(struct.pack("<B", TYPE_INT_ARRAY))
            buffer.extend(struct.pack("<I", len(val)))
            for x in val:
                buffer.extend(struct.pack("<q", x))
        elif isinstance(val, str):
            val_bytes = val.encode("utf-8")
            buffer.extend(struct.pack("<B", TYPE_STRING))
            buffer.extend(struct.pack("<I", len(val_bytes)))
            buffer.extend(val_bytes)
        else:
            raise TypeError(f"Unsupported type for field {key}: {type(val)}")
    return bytes(buffer)


def deserialize_fields(data_bytes):
    offset = 0
    num_fields = struct.unpack_from("<I", data_bytes, offset)[0]
    offset += 4
    res = {}
    for _ in range(num_fields):
        klen = struct.unpack_from("<H", data_bytes, offset)[0]
        offset += 2
        key = data_bytes[offset:offset + klen].decode("utf-8")
        offset += klen

        vtype = struct.unpack_from("<B", data_bytes, offset)[0]
        offset += 1

        if vtype == TYPE_FLOAT:
            val = struct.unpack_from("<f", data_bytes, offset)[0]
            offset += 4
            val = round(val, 6)
        elif vtype == TYPE_INT:
            val = struct.unpack_from("<q", data_bytes, offset)[0]
            offset += 8
        elif vtype == TYPE_INT_ARRAY:
            alen = struct.unpack_from("<I", data_bytes, offset)[0]
            offset += 4
            val = list(struct.unpack_from(f"<{alen}q", data_bytes, offset))
            offset += 8 * alen
        elif vtype == TYPE_STRING:
            slen = struct.unpack_from("<I", data_bytes, offset)[0]
            offset += 4
            val = data_bytes[offset:offset + slen].decode("utf-8")
            offset += slen
        else:
            raise ValueError(f"Unknown type tag {vtype}")
        res[key] = val
    return res
