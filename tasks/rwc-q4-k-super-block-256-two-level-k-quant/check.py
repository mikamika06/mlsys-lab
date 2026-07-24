import numpy as np


def _pack_6bit(values):
    out = np.zeros(12, dtype=np.uint8)
    acc = 0
    bits = 0
    pos = 0
    for v in values:
        acc |= (int(v) & 63) << bits
        bits += 6
        while bits >= 8:
            out[pos] = acc & 255
            pos += 1
            acc >>= 8
            bits -= 8
    return out


def _unpack_6bit(values):
    out = []
    acc = 0
    bits = 0
    for b in values:
        acc |= int(b) << bits
        bits += 8
        while bits >= 6 and len(out) < 16:
            out.append(acc & 63)
            acc >>= 6
            bits -= 6
    return np.array(out, dtype=np.uint8)


def _oracle_quantize(x):
    x = np.asarray(x, dtype=np.float32)
    rows, cols = x.shape
    codes = np.zeros((rows, cols // 2), dtype=np.uint8)
    sm = np.zeros((rows, cols // 256 * 12), dtype=np.uint8)
    d = np.zeros((rows, cols // 256), dtype=np.float16)
    dmin = np.zeros((rows, cols // 256), dtype=np.float16)

    for r in range(rows):
        for sb in range(cols // 256):
            block = x[r, sb * 256:(sb + 1) * 256]
            ss = []
            mm = []
            mins = []
            maxs = []
            for i in range(8):
                sub = block[i * 32:(i + 1) * 32]
                mn = float(np.min(sub))
                mx = float(np.max(sub))
                ss.append((mx - mn) / 63.0)
                mm.append(-mn / 63.0)
                mins.append(mn)
                maxs.append(mx)
            ds = max(ss)
            dm = max(mm)
            d[r, sb] = np.float16(ds / 1.0)
            dmin[r, sb] = np.float16(dm / 1.0)
            scale_codes = []
            min_codes = []
            for i in range(8):
                sc = 0 if ds == 0 else int(np.clip(round(ss[i] / ds * 63), 0, 63))
                mc = 0 if dm == 0 else int(np.clip(round(mm[i] / dm * 63), 0, 63))
                scale_codes.append(sc)
                min_codes.append(mc)
                sub = block[i * 32:(i + 1) * 32]
                step = float(d[r, sb]) * sc
                off = float(dmin[r, sb]) * mc
                q = np.zeros(32, dtype=np.uint8) if step == 0 else np.clip(
                    np.round((sub + off) / step), 0, 15
                ).astype(np.uint8)
                base = r * (cols // 2) + (sb * 128 + i * 16)
                codes.flat[base:base + 16] = q[::2] | (q[1::2] << 4)
            sm[r, sb * 12:(sb + 1) * 12] = _pack_6bit(scale_codes + min_codes)
    return codes, sm, d, dmin


def _oracle_dequant(codes, sm, d, dmin, cols):
    rows = codes.shape[0]
    out = np.zeros((rows, cols), dtype=np.float32)
    for r in range(rows):
        for sb in range(cols // 256):
            vals = _unpack_6bit(sm[r, sb * 12:(sb + 1) * 12])
            for i in range(8):
                sc = vals[i]
                mc = vals[8 + i]
                raw = codes[r, sb * 128 + i * 16:sb * 128 + i * 16 + 16]
                q = np.empty(32, dtype=np.uint8)
                q[::2] = raw & 15
                q[1::2] = raw >> 4
                out[r, sb * 256 + i * 32:sb * 256 + i * 32 + 32] = (
                    float(d[r, sb]) * sc * q.astype(np.float32)
                    - float(dmin[r, sb]) * mc
                )
    return out


def grade(sol, fx) -> dict:
    x = np.array(
        [
            np.sin(np.arange(256, dtype=np.float32) / 11.0) * 4,
            np.linspace(-8, 7, 256, dtype=np.float32),
            np.concatenate(
                [
                    np.full(64, -3, dtype=np.float32),
                    np.linspace(-1, 5, 64, dtype=np.float32),
                    np.linspace(5, -6, 64, dtype=np.float32),
                    np.zeros(64, dtype=np.float32),
                ]
            ),
        ]
    )

    ref_codes, ref_sm, ref_d, ref_dm = _oracle_quantize(x)
    try:
        codes, sm, d, dm = sol.q4k_quantize_row(x)
        recon = sol.q4k_dequantize_row(codes, sm, d, dm)
    except Exception:
        return {"max_abs_err": 1e9, "packed_exact": 0.0}

    ref_recon = _oracle_dequant(ref_codes, ref_sm, ref_d, ref_dm, x.shape[1])
    err = float(np.max(np.abs(np.asarray(recon, dtype=np.float32) - ref_recon)))
    packed = float(
        np.array_equal(np.asarray(codes, dtype=np.uint8), ref_codes)
        and np.array_equal(np.asarray(sm, dtype=np.uint8), ref_sm)
    )
    return {"max_abs_err": err, "packed_exact": packed}
