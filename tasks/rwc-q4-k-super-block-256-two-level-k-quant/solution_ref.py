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
                mx = float(sub[0])
                mn = float(sub[0])
                for val in sub:
                    v_f = float(val)
                    if v_f > mx:
                        mx = v_f
                    if v_f < mn:
                        mn = v_f
                ss.append((mx - mn) / 63)
                mm.append(-mn / 63)
            ds = ss[0]
            for val in ss:
                if val > ds:
                    ds = val
            dm = mm[0]
            for val in mm:
                if val > dm:
                    dm = val
            d[r, b] = ds
            dmin[r, b] = dm
            scs = []
            mcs = []
            for i in range(8):
                if ds == 0:
                    sc = 0
                else:
                    rc = round(ss[i] / ds * 63)
                    if rc < 0:
                        sc = 0
                    elif rc > 63:
                        sc = 63
                    else:
                        sc = int(rc)
                if dm == 0:
                    mc = 0
                else:
                    rc = round(mm[i] / dm * 63)
                    if rc < 0:
                        mc = 0
                    elif rc > 63:
                        mc = 63
                    else:
                        mc = int(rc)
                scs.append(sc)
                mcs.append(mc)
                sub = block[i * 32:(i + 1) * 32]
                step = float(d[r, b]) * sc
                off = float(dmin[r, b]) * mc
                if step == 0:
                    q = [0] * 32
                else:
                    q = []
                    for val in sub:
                        rq = round((float(val) + off) / step)
                        if rq < 0:
                            q.append(0)
                        elif rq > 15:
                            q.append(15)
                        else:
                            q.append(int(rq))
                packed_bytes = []
                for j in range(16):
                    low = q[2 * j]
                    high = q[2 * j + 1]
                    packed_bytes.append(int(low | (high << 4)))
                codes[r, b * 128 + i * 16:b * 128 + i * 16 + 16] = packed_bytes
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
                q = []
                for bval in qbytes:
                    q.append(int(bval) & 15)
                    q.append(int(bval) >> 4)
                sub_out = []
                scale_val = vals[i]
                min_val = vals[8 + i]
                d_val = float(d[r, b])
                dmin_val = float(dmin[r, b])
                for qi in q:
                    sub_out.append(d_val * scale_val * qi - dmin_val * min_val)
                out[r, b * 256 + i * 32:b * 256 + i * 32 + 32] = sub_out
    return out
