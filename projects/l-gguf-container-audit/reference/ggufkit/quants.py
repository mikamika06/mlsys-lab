import struct

QK_K = 256


def half_to_float(bits):
    sign = -1.0 if bits >> 15 else 1.0
    exp = (bits >> 10) & 0x1F
    frac = bits & 0x3FF
    if exp == 0:
        if frac == 0:
            return sign * 0.0
        return sign * frac * 2.0 ** -24
    if exp == 31:
        return sign * (float("inf") if frac == 0 else float("nan"))
    return sign * (1.0 + frac / 1024.0) * 2.0 ** (exp - 15)


def _half_at(block, off):
    return half_to_float(struct.unpack_from("<H", block, off)[0])


def _scale_min_k4(j, scales):
    if j < 4:
        return scales[j] & 63, scales[j + 4] & 63
    d = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4)
    m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4)
    return d, m


def dequant_q4_k(block):
    if len(block) != 144:
        raise ValueError("Q4_K block is 144 bytes, got %d" % len(block))
    d = _half_at(block, 0)
    dmin = _half_at(block, 2)
    scales = block[4:16]
    qs = block[16:144]
    out = [0.0] * QK_K
    at = 0
    is_ = 0
    for pair in range(4):
        sc, m = _scale_min_k4(is_, scales)
        d1, m1 = d * sc, dmin * m
        sc, m = _scale_min_k4(is_ + 1, scales)
        d2, m2 = d * sc, dmin * m
        base = pair * 32
        for l in range(32):
            b = qs[base + l]
            out[at + l] = d1 * (b & 0xF) - m1
            out[at + 32 + l] = d2 * (b >> 4) - m2
        at += 64
        is_ += 2
    return out


def dequant_q6_k(block):
    if len(block) != 210:
        raise ValueError("Q6_K block is 210 bytes, got %d" % len(block))
    ql = block[0:128]
    qh = block[128:192]
    scales = struct.unpack_from("<16b", block, 192)
    d = _half_at(block, 208)
    out = [0.0] * QK_K
    for n in range(2):
        yb = n * 128
        lb = n * 64
        hb = n * 32
        sb = n * 8
        for l in range(32):
            i = l // 16
            h = qh[hb + l]
            q1 = ((ql[lb + l] & 0xF) | (((h >> 0) & 3) << 4)) - 32
            q2 = ((ql[lb + l + 32] & 0xF) | (((h >> 2) & 3) << 4)) - 32
            q3 = ((ql[lb + l] >> 4) | (((h >> 4) & 3) << 4)) - 32
            q4 = ((ql[lb + l + 32] >> 4) | (((h >> 6) & 3) << 4)) - 32
            out[yb + l] = d * scales[sb + i] * q1
            out[yb + l + 32] = d * scales[sb + i + 2] * q2
            out[yb + l + 64] = d * scales[sb + i + 4] * q3
            out[yb + l + 96] = d * scales[sb + i + 6] * q4
    return out


def dequant_f32(block):
    return list(struct.unpack("<%df" % (len(block) // 4), block))


def dequant_f16(block):
    return [half_to_float(u) for u in
            struct.unpack("<%dH" % (len(block) // 2), block)]


BLOCK_BYTES = {0: 4, 1: 2, 12: 144, 14: 210}
BLOCK_ELEMS = {0: 1, 1: 1, 12: 256, 14: 256}
_ONE = {0: dequant_f32, 1: dequant_f16, 12: dequant_q4_k, 14: dequant_q6_k}


def dequant_tensor(raw, type_id, n_elements):
    if type_id not in _ONE:
        raise ValueError("no dequantiser for ggml type %d" % type_id)
    step = BLOCK_BYTES[type_id]
    per = BLOCK_ELEMS[type_id]
    if type_id in (0, 1):
        return _ONE[type_id](raw[:n_elements * step])
    nblocks = n_elements // per
    if len(raw) < nblocks * step:
        raise ValueError("need %d bytes for %d elements, have %d"
                         % (nblocks * step, n_elements, len(raw)))
    out = []
    fn = _ONE[type_id]
    for b in range(nblocks):
        out.extend(fn(raw[b * step:(b + 1) * step]))
    return out
