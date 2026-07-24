import numpy as np


def _pack(vals):
    out = np.zeros(12, dtype=np.uint8)
    acc = 0
    bits = 0
    p = 0
    for v in vals:
        acc |= (int(v) & 63) << bits
        bits += 6
        while bits >= 8:
            out[p] = acc & 255
            p += 1
            acc >>= 8
            bits -= 8
    return out


def _unpack(vals):
    out = []
    acc = 0
    bits = 0
    for b in vals:
        acc |= int(b) << bits
        bits += 8
        while bits >= 6 and len(out) < 16:
            out.append(acc & 63)
            acc >>= 6
            bits -= 6
    return out


def q4k_quantize_row(x):
    x = np.asarray(x, dtype=np.float32)
    rows, cols = x.shape
    codes = np.zeros((rows, cols // 2), dtype=np.uint8)
    sm = np.zeros((rows, cols // 256 * 12), dtype=np.uint8)
    d = np.zeros((rows, cols // 256), dtype=np.float16)
    dmin = np.zeros((rows, cols // 256), dtype=np.float16)

    for r in range(rows):
        for b in range(cols // 256):
            block = x[r, b * 256:(b + 1) * 256]
            ss = []
            mm = []
            for i in range(8):
                sub = block[i * 32:(i + 1) * 32]
                ss.append((float(sub.max()) - float(sub.min())) / 63)
                mm.append(-float(sub.min()) / 63)
            ds = max(ss)
            dm = max(mm)
            d[r, b] = ds
            dmin[r, b] = dm
            scs = []
            mcs = []
            for i in range(8):
                sc = 0 if ds == 0 else int(np.clip(round(ss[i] / ds * 63), 0, 63))
                mc = 0 if dm == 0 else int(np.clip(round(mm[i] / dm * 63), 0, 63))
                scs.append(sc)
                mcs.append(mc)
                sub = block[i * 32:(i + 1) * 32]
                step = float(d[r, b]) * sc
                off = float(dmin[r, b]) * mc
                q = np.zeros(32, dtype=np.uint8) if step == 0 else np.clip(
                    np.round((sub + off) / step), 0, 15
                ).astype(np.uint8)
                codes[r, b * 128 + i * 16:b * 128 + i * 16 + 16] = (
                    q[::2] | (q[1::2] << 4)
                )
            sm[r, b * 12:(b + 1) * 12] = _pack(scs + mcs)
    return codes, sm, d, dmin


def q4k_dequantize_row(codes, scales_mins, d, dmin):
    codes = np.asarray(codes, dtype=np.uint8)
    rows, packed = codes.shape
    blocks = packed // 128
    out = np.zeros((rows, blocks * 256), dtype=np.float32)
    for r in range(rows):
        for b in range(blocks):
            vals = _unpack(scales_mins[r, b * 12:(b + 1) * 12])
            for i in range(8):
                qbytes = codes[r, b * 128 + i * 16:b * 128 + i * 16 + 16]
                q = np.empty(32, dtype=np.uint8)
                q[::2] = qbytes & 15
                q[1::2] = qbytes >> 4
                out[r, b * 256 + i * 32:b * 256 + i * 32 + 32] = (
                    float(d[r, b]) * vals[i] * q
                    - float(dmin[r, b]) * vals[8 + i]
                )
    return out
