import numpy as np

from mlsys import scorers


def _oracle(W, nbits):
    qmax = (1 << (nbits - 1)) - 1
    d_out, d_in = W.shape
    per_byte = 8 // nbits
    n_bytes = d_in // per_byte

    amax = np.max(np.abs(W), axis=1)
    s = np.where(amax > 0, amax / qmax, 1.0)
    codes = np.clip(np.round(W / s[:, None]), -qmax, qmax).astype(np.int64)
    u = (codes + qmax).astype(np.uint8)  # unsigned, range [0, 2*qmax]

    packed = np.zeros((d_out, n_bytes), dtype=np.uint8)
    for k in range(per_byte):
        packed |= (u[:, k::per_byte] << (k * nbits)).astype(np.uint8)

    mask = (1 << nbits) - 1
    unpacked = np.zeros((d_out, d_in), dtype=np.uint8)
    for k in range(per_byte):
        unpacked[:, k::per_byte] = (packed >> (k * nbits)) & mask
    dequant = (unpacked.astype(np.int64) - qmax) * s[:, None]

    return packed, dequant


def grade(sol, fx) -> dict:
    """
    Runs the reference symmetric quantize + low-bit-first sub-byte pack
    (2 codes/byte for nbits=4, 4 codes/byte for nbits=2) with a NumPy
    oracle on the fixed weight fixture, for both bit widths. Compares the
    submission's packed uint8 buffer (byte-exact) and its dequantized
    reconstruction (max abs error) to the oracle's, for each bit width,
    and reports the worst case across both.
    """
    W = np.asarray(fx["qnt_w"], dtype=np.float64)

    byte_frac_worst = 1.0
    dequant_err_worst = 0.0
    for nbits in (4, 2):
        packed_exp, dequant_exp = _oracle(W, nbits)
        try:
            packed_got, _s_got, dequant_got = sol.pack_sub_byte(W.copy(), nbits)
            packed_got = np.asarray(packed_got, dtype=np.uint8)
            dequant_got = np.asarray(dequant_got, dtype=np.float64)
        except Exception:
            return {"byte_exact_fraction": 0.0, "dequant_max_abs_err": float("inf")}

        if packed_got.shape != packed_exp.shape:
            byte_frac = 0.0
        else:
            byte_frac = scorers.byte_exact_fraction(packed_exp, packed_got)
        byte_frac_worst = min(byte_frac_worst, byte_frac)

        if dequant_got.shape != dequant_exp.shape:
            dequant_err = float("inf")
        else:
            dequant_err = scorers.max_abs_err(dequant_exp, dequant_got)
        dequant_err_worst = max(dequant_err_worst, dequant_err)

    return {"byte_exact_fraction": byte_frac_worst, "dequant_max_abs_err": dequant_err_worst}
