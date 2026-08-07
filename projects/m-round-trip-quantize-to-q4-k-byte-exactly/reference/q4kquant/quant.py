import numpy as np


def _pack_6bit_pairs(scales: np.ndarray, mins: np.ndarray) -> bytes:
    buf = bytearray(12)
    sc = scales.astype(np.uint8) & 0x3F
    mn = mins.astype(np.uint8) & 0x3F

    for chunk in range(2):
        i = chunk * 4
        s = sc[i : i + 4]
        out_offset = chunk * 3
        buf[out_offset] = s[0] | ((s[1] & 0x03) << 6)
        buf[out_offset + 1] = (s[1] >> 2) | ((s[2] & 0x0F) << 4)
        buf[out_offset + 2] = (s[2] >> 4) | (s[3] << 2)

    for chunk in range(2):
        i = chunk * 4
        m = mn[i : i + 4]
        out_offset = 6 + chunk * 3
        buf[out_offset] = m[0] | ((m[1] & 0x03) << 6)
        buf[out_offset + 1] = (m[1] >> 2) | ((m[2] & 0x0F) << 4)
        buf[out_offset + 2] = (m[2] >> 4) | (m[3] << 2)

    return bytes(buf)


def _unpack_6bit_pairs(b: bytes) -> tuple[np.ndarray, np.ndarray]:
    sc = np.zeros(8, dtype=np.uint8)
    mn = np.zeros(8, dtype=np.uint8)

    for chunk in range(2):
        out_offset = chunk * 3
        b0, b1, b2 = b[out_offset], b[out_offset + 1], b[out_offset + 2]
        i = chunk * 4
        sc[i] = b0 & 0x3F
        sc[i + 1] = (b0 >> 6) | ((b1 & 0x0F) << 2)
        sc[i + 2] = (b1 >> 4) | ((b2 & 0x03) << 4)
        sc[i + 3] = b2 >> 2

    for chunk in range(2):
        out_offset = 6 + chunk * 3
        b0, b1, b2 = b[out_offset], b[out_offset + 1], b[out_offset + 2]
        i = chunk * 4
        mn[i] = b0 & 0x3F
        mn[i + 1] = (b0 >> 6) | ((b1 & 0x0F) << 2)
        mn[i + 2] = (b1 >> 4) | ((b2 & 0x03) << 4)
        mn[i + 3] = b2 >> 2

    return sc, mn


def quantize_q4_k(x: np.ndarray) -> bytes:
    x_flat = np.asarray(x, dtype=np.float32).ravel()
    n_blocks = len(x_flat) // 256
    out = bytearray()

    for sb in range(n_blocks):
        block = x_flat[sb * 256 : (sb + 1) * 256]
        subblocks = block.reshape(8, 32)

        sub_mins = np.min(subblocks, axis=1)
        sub_maxs = np.max(subblocks, axis=1)
        sub_ranges = sub_maxs - sub_mins

        max_range = float(np.max(sub_ranges))
        d_val = max_range / 63.0 if max_range > 1e-10 else 1.0
        d_fp16 = np.float16(d_val)
        d_val = float(d_fp16) if float(d_fp16) != 0.0 else 1.0

        max_min = float(np.max(np.abs(sub_mins)))
        dmin_val = max_min / 63.0 if max_min > 1e-10 else 1.0
        dmin_fp16 = np.float16(dmin_val)
        dmin_val = float(dmin_fp16) if float(dmin_fp16) != 0.0 else 1.0

        sc_quant = np.clip(np.round(sub_ranges / d_val), 0, 63).astype(np.uint8)
        mn_quant = np.clip(np.round(np.abs(sub_mins) / dmin_val), 0, 63).astype(np.uint8)

        eff_scales = sc_quant.astype(np.float32) * d_val
        eff_mins = mn_quant.astype(np.float32) * dmin_val

        qs = np.zeros((8, 32), dtype=np.uint8)
        for i in range(8):
            s = eff_scales[i] if eff_scales[i] > 1e-10 else 1.0
            m = eff_mins[i]
            q = np.clip(np.round((subblocks[i] + m) / s * 15.0), 0, 15).astype(np.uint8)
            qs[i] = q

        qs_flat = qs.ravel()
        qs_packed = bytearray(128)
        for i in range(128):
            qs_packed[i] = qs_flat[2 * i] | (qs_flat[2 * i + 1] << 4)

        out.extend(np.float16(d_val).tobytes())
        out.extend(np.float16(dmin_val).tobytes())
        out.extend(_pack_6bit_pairs(sc_quant, mn_quant))
        out.extend(qs_packed)

    return bytes(out)


def dequantize_q4_k(b: bytes, shape: tuple) -> np.ndarray:
    n_elements = int(np.prod(shape))
    n_blocks = n_elements // 256
    out = np.zeros(n_elements, dtype=np.float32)

    for sb in range(n_blocks):
        offset = sb * 144
        d = float(np.frombuffer(b[offset : offset + 2], dtype=np.float16)[0])
        dmin = float(np.frombuffer(b[offset + 2 : offset + 4], dtype=np.float16)[0])

        sc, mn = _unpack_6bit_pairs(b[offset + 4 : offset + 16])
        eff_scales = sc.astype(np.float32) * d
        eff_mins = mn.astype(np.float32) * dmin

        qs_raw = np.frombuffer(b[offset + 16 : offset + 144], dtype=np.uint8)
        qs = np.zeros(256, dtype=np.float32)
        qs[0::2] = (qs_raw & 0x0F).astype(np.float32)
        qs[1::2] = (qs_raw >> 4).astype(np.float32)

        qs_sub = qs.reshape(8, 32)
        dequant = np.zeros((8, 32), dtype=np.float32)
        for i in range(8):
            s = eff_scales[i] if eff_scales[i] > 1e-10 else 1.0
            m = eff_mins[i]
            dequant[i] = (qs_sub[i] / 15.0) * s - m

        out[sb * 256 : (sb + 1) * 256] = dequant.ravel()

    return out.reshape(shape)


def quantize_q4_0(x: np.ndarray) -> bytes:
    x_flat = np.asarray(x, dtype=np.float32).ravel()
    n_blocks = len(x_flat) // 32
    out = bytearray()

    for b in range(n_blocks):
        block = x_flat[b * 32 : (b + 1) * 32]
        max_abs = float(np.max(np.abs(block)))
        d_val = max_abs / 7.0 if max_abs > 1e-10 else 1.0
        d_fp16 = np.float16(d_val)
        d_val = float(d_fp16) if float(d_fp16) != 0.0 else 1.0

        q = np.clip(np.round(block / d_val) + 8, 0, 15).astype(np.uint8)
        qs_packed = bytearray(16)
        for i in range(16):
            qs_packed[i] = q[2 * i] | (q[2 * i + 1] << 4)

        out.extend(np.float16(d_val).tobytes())
        out.extend(qs_packed)

    return bytes(out)


def dequantize_q4_0(b: bytes, shape: tuple) -> np.ndarray:
    n_elements = int(np.prod(shape))
    n_blocks = n_elements // 32
    out = np.zeros(n_elements, dtype=np.float32)

    for b_idx in range(n_blocks):
        offset = b_idx * 18
        d = float(np.frombuffer(b[offset : offset + 2], dtype=np.float16)[0])

        qs_raw = np.frombuffer(b[offset + 2 : offset + 18], dtype=np.uint8)
        q = np.zeros(32, dtype=np.float32)
        q[0::2] = (qs_raw & 0x0F).astype(np.float32)
        q[1::2] = (qs_raw >> 4).astype(np.float32)

        out[b_idx * 32 : (b_idx + 1) * 32] = (q - 8.0) * d

    return out.reshape(shape)
