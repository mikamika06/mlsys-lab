import numpy as np


def q4k_quantize_superblock(x):
    """Q4_K two-level (super-block) asymmetric quantization.

    `x` is a float32 array of shape (rows, cols) with cols a multiple of
    256. Each row is split into super-blocks of 256 values, and each
    super-block into 8 sub-blocks of 32 values.

    For sub-block i within a super-block:
        mn_i, mx_i = min(sub_i), max(sub_i)
        ss_i = (mx_i - mn_i) / 63        # sub-scale, in value units
        mm_i = -mn_i / 63                # sub-min,   in value units

    The super-block scale and min are the largest magnitude of the 8
    sub-scales / sub-mins:
        d    = max_i(ss_i)
        dmin = max_i(mm_i)

    Each sub-scale and sub-min is then re-quantized to a 6-bit code in
    [0, 63] relative to its super-block value (`d`, `dmin`), and every
    weight is quantized to a 4-bit code:
        sc_i = round(ss_i / d * 63),  mc_i = round(mm_i / dmin * 63)
        step = d * sc_i,  off = dmin * mc_i
        q    = clip(round((w + off) / step), 0, 15)

    Returns
    -------
    codes : uint8, shape (rows, cols // 2)
        4-bit codes packed two per byte (low nibble = even index, high
        nibble = odd index within each 32-element sub-block).
    sub_scales : uint8, shape (rows, cols // 256, 8)
        The 8 per-sub-block 6-bit scale codes, one super-block row per
        entry along axis 1.
    sub_mins : uint8, shape (rows, cols // 256, 8)
        The 8 per-sub-block 6-bit min codes.
    d : float16, shape (rows, cols // 256)
        Super-block scale.
    dmin : float16, shape (rows, cols // 256)
        Super-block min scale.
    """
    x = np.asarray(x, dtype=np.float32)
    rows, cols = x.shape
    n_sb = cols // 256

    codes = np.zeros((rows, cols // 2), dtype=np.uint8)
    sub_scales = np.zeros((rows, n_sb, 8), dtype=np.uint8)
    sub_mins = np.zeros((rows, n_sb, 8), dtype=np.uint8)
    d = np.zeros((rows, n_sb), dtype=np.float16)
    dmin = np.zeros((rows, n_sb), dtype=np.float16)

    for r in range(rows):
        for sb in range(n_sb):
            block = x[r, sb * 256:(sb + 1) * 256]
            ss = []
            mm = []
            for i in range(8):
                sub = block[i * 32:(i + 1) * 32]
                mn = float(sub[0])
                mx = float(sub[0])
                for val in sub:
                    v = float(val)
                    if v < mn:
                        mn = v
                    if v > mx:
                        mx = v
                ss.append((mx - mn) / 63.0)
                mm.append(-mn / 63.0)

            ds = ss[0]
            for val in ss:
                if val > ds:
                    ds = val
            dm = mm[0]
            for val in mm:
                if val > dm:
                    dm = val

            d[r, sb] = np.float16(ds)
            dmin[r, sb] = np.float16(dm)

            for i in range(8):
                sc = 0 if ds == 0 else int(max(0, min(63, round(ss[i] / ds * 63))))
                mc = 0 if dm == 0 else int(max(0, min(63, round(mm[i] / dm * 63))))
                sub_scales[r, sb, i] = sc
                sub_mins[r, sb, i] = mc

                sub = block[i * 32:(i + 1) * 32]
                step = float(d[r, sb]) * sc
                off = float(dmin[r, sb]) * mc
                
                q_list = []
                if step == 0:
                    for _ in range(32):
                        q_list.append(0)
                else:
                    for val in sub:
                        val_f = float(val)
                        r_val = round((val_f + off) / step)
                        c_val = max(0, min(15, r_val))
                        q_list.append(int(c_val))
                
                base = sb * 128 + i * 16
                for idx_byte in range(16):
                    even_q = q_list[idx_byte * 2]
                    odd_q = q_list[idx_byte * 2 + 1]
                    codes[r, base + idx_byte] = even_q | (odd_q << 4)

    return codes, sub_scales, sub_mins, d, dmin


def q4k_dequantize_superblock(codes, sub_scales, sub_mins, d, dmin):
    """Inverse of `q4k_quantize_superblock`.

    w_hat = d[sb] * sub_scales[sb, i] * q  -  dmin[sb] * sub_mins[sb, i]

    for every sub-block i (32 values) inside every super-block sb.

    Returns
    -------
    out : float32, shape (rows, (packed_cols_per_row * 2))
    """
    codes = np.asarray(codes, dtype=np.uint8)
    sub_scales = np.asarray(sub_scales)
    sub_mins = np.asarray(sub_mins)
    rows, packed = codes.shape
    n_sb = packed // 128
    out = np.zeros((rows, n_sb * 256), dtype=np.float32)

    for r in range(rows):
        for sb in range(n_sb):
            dv = float(d[r, sb])
            dmv = float(dmin[r, sb])
            for i in range(8):
                sc = int(sub_scales[r, sb, i])
                mc = int(sub_mins[r, sb, i])
                qbytes = codes[r, sb * 128 + i * 16:sb * 128 + i * 16 + 16]
                out_vals = []
                for idx_byte in range(16):
                    b_val = int(qbytes[idx_byte])
                    q_even = b_val & 15
                    q_odd = b_val >> 4
                    val_even = dv * sc * float(q_even) - dmv * mc
                    val_odd = dv * sc * float(q_odd) - dmv * mc
                    out_vals.append(val_even)
                    out_vals.append(val_odd)
                out[r, sb * 256 + i * 32:sb * 256 + i * 32 + 32] = np.asarray(out_vals, dtype=np.float32)
    return out
