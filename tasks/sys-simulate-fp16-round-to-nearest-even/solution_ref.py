import numpy as np


def fp32_to_fp16_rne(x: np.ndarray) -> np.ndarray:
    bits = x.view(np.uint32)
    out = np.empty(bits.shape, dtype=np.uint16)

    flat_in = bits.ravel()
    flat_out = out.ravel()

    for i, v in enumerate(flat_in):
        sign = (v >> 31) & 1
        exp = (v >> 23) & 0xff
        frac = v & 0x7fffff

        if exp == 0xff:
            if frac == 0:
                half = (sign << 15) | (0x1f << 10)
            else:
                half = (sign << 15) | (0x1f << 10) | 0x200
            flat_out[i] = np.uint16(half)
            continue

        new_exp = int(exp) - 127 + 15

        if new_exp >= 31:
            flat_out[i] = np.uint16((sign << 15) | (0x1f << 10))
            continue

        if new_exp <= 0:
            if new_exp < -10:
                flat_out[i] = np.uint16(sign << 15)
                continue

            mant = frac | 0x800000
            shift = 14 - new_exp
            kept = mant >> shift
            rem = mant & ((1 << shift) - 1)
            half = 1 << (shift - 1)

            if rem > half or (rem == half and (kept & 1)):
                kept += 1

            flat_out[i] = np.uint16((sign << 15) | kept)
            continue

        kept = frac >> 13
        rem = frac & 0x1fff

        if rem > 0x1000 or (rem == 0x1000 and (kept & 1)):
            kept += 1
            if kept == 0x400:
                new_exp += 1
                kept = 0
                if new_exp >= 31:
                    flat_out[i] = np.uint16((sign << 15) | (0x1f << 10))
                    continue

        flat_out[i] = np.uint16(
            (sign << 15) | (new_exp << 10) | kept
        )

    return out.view(np.float16)
