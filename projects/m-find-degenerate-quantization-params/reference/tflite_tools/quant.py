import struct

def find_degenerate_quantization_params(fb_bytes):
    degenerate = []
    idx = 4
    tensor_idx = 0
    while idx < len(fb_bytes):
        if idx + 4 > len(fb_bytes):
            break
        name_len = struct.unpack("<I", fb_bytes[idx:idx+4])[0]
        idx += 4
        if idx + name_len > len(fb_bytes):
            break
        idx += name_len
        if idx + 8 > len(fb_bytes):
            break
        scale, zp = struct.unpack("<fI", fb_bytes[idx:idx+8])
        idx += 8
        if scale <= 0.0:
            degenerate.append(tensor_idx)
        tensor_idx += 1
    return degenerate
