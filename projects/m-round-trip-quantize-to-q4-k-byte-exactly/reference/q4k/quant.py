import numpy as np


def _pack_scales_and_mins(scales: np.ndarray, mins: np.ndarray) -> bytes:
    scales_u64 = np.zeros(3, dtype=np.uint64)
    mins_u64 = np.zeros(3, dtype=np.uint64)
    for i in range(8):
        sc = int(scales[i]) & 0x3F
        mn = int(mins[i]) & 0x3F
        scales_u64[i // 3] |= np.uint64(sc) << np.uint64(8 * (i % 3))
        mins_u64[i // 3] |= np.uint64(mn) << np.uint64(8 * (i % 3))
    for i in range(8, 16):
        sc = int(scales[i]) & 0x3F
        mn = int(mins[i]) & 0x3F
        shift = 8 * ((i - 8) % 3)
        idx = 2 + (i - 8) // 3
        if idx < 3:
            scales_u64[idx] |= np.uint64(sc) << np.uint64(shift + 24)
            mins_u64[idx] |= np.uint64(mn) << np.uint64(shift + 24)
    buf = bytearray()
    for v in scales_u64:
        buf.extend(int(v).to_bytes(8, 'little'))
    for v in mins_u64:
        buf.extend(int(v).to_bytes(8, 'little'))
    return bytes(buf)


def _unpack_scales_and_mins(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    scales = np.zeros(16, dtype=np.uint8)
    mins = np.zeros(16, dtype=np.uint8)
    scales_u64 = [int.from_bytes(data[i:i+8], 'little') for i in range(0, 24, 8)]
    mins_u64 = [int.from_bytes(data[i:i+8], 'little') for i in range(24, 48, 8)]
    for i in range(8):
        val_sc = (scales_u64[i // 3] >> (8 * (i % 3))) & 0x3F
        val_mn = (mins_u64[i // 3] >> (8 * (i % 3))) & 0x3F
        scales[i] = np.uint8(val_sc)
        mins[i] = np.uint8(val_mn)
    for i in range(8, 16):
        shift = 8 * ((i - 8) % 3)
        idx = 2 + (i - 8) // 3
        if idx < 3:
            val_sc = (scales_u64[idx] >> (shift + 24)) & 0x3F
            val_mn = (mins_u64[idx] >> (shift + 24)) & 0x3F
            scales[i] = np.uint8(val_sc)
            mins[i] = np.uint8(val_mn)
    return scales, mins


def quantize_q4_k_superblock(weights: np.ndarray) -> bytes:
    w = weights.astype(np.float32).flatten()
    assert len(w) == 256
    d = np.float16(np.max(np.abs(w)) / 15.0).astype(np.float32)
    d_fp16 = np.float16(d).view(np.uint16).item()
    scales = np.zeros(16, dtype=np.uint8)
    mins = np.zeros(16, dtype=np.uint8)
    sub_quants = np.zeros((16, 32), dtype=np.uint8)
    for i in range(16):
        sub = w[i * 16:(i + 1) * 16]
        mx = np.max(sub)
        mn = np.min(sub)
        scale_val = (mx - mn) / 15.0 if mx > mn else 0.0
        sc = int(np.round(scale_val * 10.0)) & 0x3F
        mn_val = int(np.round(mn * 10.0)) & 0x3F
        scales[i] = sc
        mins[i] = mn_val
        eff_scale = sc / 10.0 if sc > 0 else 1.0
        eff_min = mn_val / 10.0
        q32 = np.clip(np.round((sub - eff_min) / eff_scale), 0, 15).astype(np.uint8)
        for j in range(16):
            low = q32[j] & 0x0F
            high = q32[j + 16] & 0x0F
            sub_quants[i, j] = low | (high << 4)
    d_bytes = int(d_fp16).to_bytes(2, 'little')
    dmin_bytes = b'\x00\x00'
    sm_bytes = _pack_scales_and_mins(scales, mins)
    ql_bytes = bytearray()
    for i in range(16):
        ql_bytes.extend(sub_quants[i].tobytes())
    return d_bytes + dmin_bytes + sm_bytes + bytes(ql_bytes)


def dequantize_q4_k_superblock(data: bytes) -> np.ndarray:
    d_fp16 = int.from_bytes(data[0:2], 'little')
    d = float(np.float16(np.array([d_fp16], dtype=np.uint16).view(np.float16)[0]))
    sm_data = data[4:52]
    scales, mins = _unpack_scales_and_mins(sm_data)
    ql_data = data[52:]
    out = np.zeros(256, dtype=np.float32)
    for i in range(16):
        sc = float(scales[i]) / 10.0
        mn = float(mins[i]) / 10.0
        sub_bytes = ql_data[i * 16:(i + 1) * 16]
        for j in range(16):
            b = sub_bytes[j]
            low = b & 0x0F
            high = (b >> 4) & 0x0F
            out[i * 16 + j] = float(low) * sc + mn
            out[i * 16 + 16 + j] = float(high) * sc + mn
    return out


def round_trip_q4_k(weights: np.ndarray) -> tuple[bytes, np.ndarray]:
    b = quantize_q4_k_superblock(weights)
    dec = dequantize_q4_k_superblock(b)
    return b, dec
