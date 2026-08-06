import struct
import random

def generate_q4_k_block(seed):
    random.seed(seed)
    d = random.uniform(0.1, 1.0)
    dmin = random.uniform(0.01, 0.5)
    scales_bytes = bytes([random.randint(0, 255) for _ in range(12)])
    qs = bytes([random.randint(0, 255) for _ in range(128)])
    return struct.pack('<ee', d, dmin) + scales_bytes + qs

def generate_q6_k_block(seed):
    random.seed(seed)
    ql = bytes([random.randint(0, 255) for _ in range(128)])
    qh = bytes([random.randint(0, 255) for _ in range(64)])
    scales = bytes([random.randint(0, 255) for _ in range(16)])
    d = random.uniform(0.1, 1.0)
    return ql + qh + scales + struct.pack('<e', d)

def unpack_6bit_scales_and_mins(q: bytes) -> tuple[list[int], list[int]]:
    scales = [0] * 8
    mins = [0] * 8
    for j in range(4):
        scales[j] = q[j] & 63
        mins[j] = q[j + 4] & 63
    for j in range(4, 8):
        scales[j] = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)
        mins[j] = (q[j + 4] >> 4) | ((q[j] >> 6) << 4)
    return scales, mins

def dequantize_q4_k(block: bytes) -> list[float]:
    d, dmin = struct.unpack('<ee', block[:4])
    scales, mins = unpack_6bit_scales_and_mins(block[4:16])
    qs = block[16:144]
    y = [0.0] * 256
    for b in range(8):
        scale = scales[b]
        m = mins[b]
        offset = (b // 2) * 32
        for l in range(32):
            if b % 2 == 0:
                q_val = qs[offset + l] & 0xF
            else:
                q_val = qs[offset + l] >> 4
            y[b * 32 + l] = q_val * (d * scale) - (dmin * m)
    return y

def dequantize_q6_k(block: bytes) -> list[float]:
    ql = block[:128]
    qh = block[128:192]
    scales = struct.unpack('<16b', block[192:208])
    d = struct.unpack('<e', block[208:210])[0]
    y = [0.0] * 256
    for l in range(64):
        q1 = (ql[l] & 0xF) | ((qh[l] & 3) << 4)
        q2 = (ql[l] >> 4) | (((qh[l] >> 2) & 3) << 4)
        q3 = (ql[l + 64] & 0xF) | (((qh[l] >> 4) & 3) << 4)
        q4 = (ql[l + 64] >> 4) | (((qh[l] >> 6) & 3) << 4)
        y[l] = d * scales[l // 16] * (q1 - 32)
        y[l + 64] = d * scales[l // 16 + 4] * (q2 - 32)
        y[l + 128] = d * scales[l // 16 + 8] * (q3 - 32)
        y[l + 192] = d * scales[l // 16 + 12] * (q4 - 32)
    return y

Q4_FIXTURES = [generate_q4_k_block(i) for i in range(10)]
Q6_FIXTURES = [generate_q6_k_block(i) for i in range(10)]
