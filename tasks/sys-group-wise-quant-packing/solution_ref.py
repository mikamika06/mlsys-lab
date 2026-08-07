import struct


def _to_float32(val: float) -> float:
    return struct.unpack('<f', struct.pack('<f', float(val)))[0]


def pack_groupwise_int4(x: list[float]) -> bytes:
    out = bytearray()
    for start in range(0, len(x), 8):
        g = x[start:start + 8]
        m = _to_float32(0.0)
        for val in g:
            fval = _to_float32(val)
            abs_val = fval if fval >= 0.0 else -fval
            if abs_val > m:
                m = abs_val
        scale = _to_float32(_to_float32(m) / _to_float32(7.0)) if m != 0.0 else _to_float32(1.0)
        n = []
        for val in g:
            fval = _to_float32(val)
            r = round(float(_to_float32(fval / scale)))
            if r < -8:
                r = -8
            elif r > 7:
                r = 7
            n.append(int(r) + 8)
        out.extend(struct.pack("<f", float(scale)))
        for i in range(0, 8, 2):
            out.append(int(n[i]) | (int(n[i + 1]) << 4))
    return bytes(out)
