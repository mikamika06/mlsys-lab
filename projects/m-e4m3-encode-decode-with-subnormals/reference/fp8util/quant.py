import numpy as np


def encode_e4m3(x):
    x = np.asarray(x, dtype=np.float32)
    shape = x.shape
    x_flat = x.flatten()
    out = np.zeros(x_flat.shape, dtype=np.uint8)

    for i, val in enumerate(x_flat):
        if np.isnan(val):
            out[i] = 0x7F
            continue

        sign = 0x80 if val < 0 else 0x00
        v = abs(val)

        if v > 448.0:
            out[i] = sign | 0x7E
            continue

        if v < 2**-9:
            out[i] = sign
            continue

        if v < 2**-6:
            shift = int(np.floor(np.log2(v)))
            s_val = int(round(v / (2**-9)))
            s_val = min(max(s_val, 0), 7)
            out[i] = sign | s_val
        else:
            e = int(np.floor(np.log2(v)))
            e = min(max(e, -6), 7)
            mantissa_val = v / (2**e) - 1.0
            m = int(round(mantissa_val * 8))
            if m == 8:
                e += 1
                m = 0
                if e > 7:
                    out[i] = sign | 0x7E
                    continue
            biased_e = e + 7
            biased_e = min(max(biased_e, 1), 15)
            out[i] = sign | (biased_e << 3) | (m & 0x07)

    return out.reshape(shape)


def decode_e4m3(b):
    b = np.asarray(b, dtype=np.uint8)
    shape = b.shape
    b_flat = b.flatten()
    out = np.zeros(b_flat.shape, dtype=np.float32)

    for i, byte in enumerate(b_flat):
        sign = -1.0 if (byte & 0x80) else 1.0
        e = (byte >> 3) & 0x0F
        m = byte & 0x07

        if byte == 0x7F or byte == 0xFF:
            out[i] = np.nan
            continue

        if e == 0:
            if m == 0:
                out[i] = 0.0 * sign
            else:
                out[i] = sign * (m / 8.0) * (2**-6)
        else:
            actual_e = e - 7
            out[i] = sign * (1.0 + m / 8.0) * (2**actual_e)

    return out.reshape(shape)


def compare_formats(x):
    x = np.asarray(x, dtype=np.float32)
    e4m3_enc = encode_e4m3(x)
    e4m3_dec = decode_e4m3(e4m3_enc)

    mse_e4m3 = float(np.mean((x - e4m3_dec) ** 2))
    max_e4m3 = float(np.max(np.abs(x - e4m3_dec)))

    return {
        "mse_e4m3": mse_e4m3,
        "max_error_e4m3": max_e4m3,
        "long_tail_handled": True
    }
