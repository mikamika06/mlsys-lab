import struct

def rebuild_flatbuffer(fb_bytes, name_mapping):
    data = bytearray(b"TFL3")
    idx = 4
    while idx < len(fb_bytes):
        if idx + 4 > len(fb_bytes):
            break
        name_len = struct.unpack("<I", fb_bytes[idx:idx+4])[0]
        idx += 4
        name = fb_bytes[idx:idx+name_len].decode("utf-8")
        idx += name_len
        scale, zp = struct.unpack("<fI", fb_bytes[idx:idx+8])
        idx += 8

        new_name = name_mapping.get(name, name)
        new_name_bytes = new_name.encode("utf-8")
        data.extend(struct.pack("<I", len(new_name_bytes)))
        data.extend(new_name_bytes)
        data.extend(struct.pack("<fI", scale, zp))
    return bytes(data)
