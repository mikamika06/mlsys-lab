import math
import numpy as np


def pack_sub_byte(W: np.ndarray, nbits: int):
    """
    Per-row symmetric quantization, then pack multiple low-bit codes into
    each uint8 byte, low-order (least significant) code first:

    1. scale s = max(|row|) / qmax, qmax = 2^(nbits-1) - 1.
    2. code = clip(round(w/s), -qmax, qmax); unsigned u = code + qmax
       (fits in `nbits` bits, range [0, 2*qmax]).
    3. `per_byte = 8 // nbits` codes packed per byte:
       byte = u[0] | (u[1] << nbits) | (u[2] << 2*nbits) | ...
       (`d_in` must be divisible by `per_byte`.)
    4. Dequantization unpacks the same fields and maps back:
       w_hat = (u - qmax) * s.

    Returns (packed, s, dequant):
      packed  -- uint8 array, shape (d_out, d_in * nbits // 8).
      s       -- float array, shape (d_out,), per-row scale.
      dequant -- float array, shape (d_out, d_in), reconstruction from the
                 packed buffer (round-trip through pack + unpack).
    """
    W = np.asarray(W, dtype=np.float64)
    qmax = (1 << (nbits - 1)) - 1
    d_out, d_in = W.shape
    per_byte = 8 // nbits
    n_bytes = d_in // per_byte

    s_list = []
    packed_list = []
    dequant_list = []

    for i in range(d_out):
        row = W[i]
        
        max_abs = 0.0
        for j in range(d_in):
            val = row[j]
            if val < 0.0:
                val = -val
            if val > max_abs:
                max_abs = val
        
        if max_abs > 0.0:
            s_val = max_abs / qmax
        else:
            s_val = 1.0
        s_list.append(s_val)

        row_codes = []
        row_u = []
        for j in range(d_in):
            div = row[j] / s_val
            rounded = round(div)
            if rounded < -qmax:
                clipped = -qmax
            elif rounded > qmax:
                clipped = qmax
            else:
                clipped = rounded
            
            code = int(clipped)
            row_codes.append(code)
            u_val = code + qmax
            row_u.append(u_val)

        row_packed = []
        for b in range(n_bytes):
            byte_val = 0
            for k in range(per_byte):
                idx = b * per_byte + k
                byte_val |= (row_u[idx] << (k * nbits))
            row_packed.append(byte_val & 0xFF)
        packed_list.append(row_packed)

        row_unpacked = []
        mask = (1 << nbits) - 1
        for b in range(n_bytes):
            byte_val = row_packed[b]
            for k in range(per_byte):
                unpacked_val = (byte_val >> (k * nbits)) & mask
                row_unpacked.append(unpacked_val)

        row_dequant = []
        for j in range(d_in):
            dq = (row_unpacked[j] - qmax) * s_val
            row_dequant.append(dq)
        dequant_list.append(row_dequant)

    packed = np.array(packed_list, dtype=np.uint8)
    s = np.array(s_list, dtype=np.float64)
    dequant = np.array(dequant_list, dtype=np.float64)

    return packed, s, dequant
