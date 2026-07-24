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

    amax = np.max(np.abs(W), axis=1)
    s = np.where(amax > 0, amax / qmax, 1.0)
    codes = np.clip(np.round(W / s[:, None]), -qmax, qmax).astype(np.int64)
    u = (codes + qmax).astype(np.uint8)

    packed = np.zeros((d_out, n_bytes), dtype=np.uint8)
    for k in range(per_byte):
        packed |= (u[:, k::per_byte] << (k * nbits)).astype(np.uint8)

    mask = (1 << nbits) - 1
    unpacked = np.zeros((d_out, d_in), dtype=np.uint8)
    for k in range(per_byte):
        unpacked[:, k::per_byte] = (packed >> (k * nbits)) & mask
    dequant = (unpacked.astype(np.int64) - qmax) * s[:, None]

    return packed, s, dequant
