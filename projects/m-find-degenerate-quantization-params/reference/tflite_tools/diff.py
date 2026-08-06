import struct

def _parse_tensors(fb_bytes):
    tensors = []
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
        tensors.append({"name": name, "scale": scale, "zero_point": zp})
    return tensors

def structural_diff(fb1_bytes, fb2_bytes):
    t1 = _parse_tensors(fb1_bytes)
    t2 = _parse_tensors(fb2_bytes)
    changed = []
    for i, (a, b) in enumerate(zip(t1, t2)):
        if a != b:
            changed.append({"index": i, "old": a, "new": b})
    return {"changed_tensors": changed}
