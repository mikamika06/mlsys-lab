import numpy as np

from mlsys import scorers


def _quantize_pack(W, group_size):
    """Per-group asymmetric 4-bit quant (same scheme as GPTQ), then pack two
    4-bit codes per uint8 byte (low nibble = even raveled index, high nibble
    = odd raveled index; unused high nibble on a trailing odd-length array
    is left 0), mirroring an AffineQuantizedTensor's int_data storage."""
    shape = W.shape
    flat = np.asarray(W, dtype=np.float64).ravel()
    n = flat.size
    codes = np.empty(n, dtype=np.uint8)
    scales = []
    zeros = []
    for start in range(0, n, group_size):
        g = flat[start:start + group_size]
        gmax = float(g.max())
        gmin = float(g.min())
        span = gmax - gmin
        scale = 1.0 if span == 0.0 else span / 15.0
        zero = float(np.clip(np.rint(-gmin / scale), 0, 15))
        code = np.clip(np.rint(g / scale) + zero, 0, 15).astype(np.uint8)
        codes[start:start + len(g)] = code
        scales.append(scale)
        zeros.append(zero)

    n_bytes = (n + 1) // 2
    packed = np.zeros(n_bytes, dtype=np.uint8)
    for i in range(n_bytes):
        lo = int(codes[2 * i])
        hi = int(codes[2 * i + 1]) if 2 * i + 1 < n else 0
        packed[i] = (lo & 0x0F) | ((hi & 0x0F) << 4)

    return packed, np.asarray(scales, dtype=np.float64), np.asarray(zeros, dtype=np.float64), shape


def _dequant_dense(packed, scale, zero, group_size, shape):
    n = int(np.prod(shape))
    codes = np.empty(n, dtype=np.uint8)
    packed = np.asarray(packed, dtype=np.uint8)
    for i in range(len(packed)):
        b = int(packed[i])
        codes[2 * i] = b & 0x0F
        if 2 * i + 1 < n:
            codes[2 * i + 1] = (b >> 4) & 0x0F

    out = np.empty(n, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    zero = np.asarray(zero, dtype=np.float64)
    for gi in range(len(scale)):
        start = gi * group_size
        end = min(n, start + group_size)
        out[start:end] = (codes[start:end].astype(np.float64) - zero[gi]) * scale[gi]
    return out.reshape(shape)


def _cases():
    rng = np.random.default_rng(7)
    cases = []
    cases.append((np.array([0.0, 1.5, -2.5, 7.0, 3.0], dtype=np.float64), 2))     # odd n, odd group_size
    cases.append((rng.normal(size=(4, 5)) * 3.0, 3))                              # 2D, group crosses rows
    cases.append((rng.uniform(-5, 5, size=(3, 7)), 4))                            # odd total (21), group 4
    cases.append((np.zeros((2, 3), dtype=np.float64), 4))                         # constant group
    cases.append((rng.normal(loc=2.0, scale=1.5, size=(9,)), 8))                  # last group size 1
    return cases


def grade(sol, fx) -> dict:
    worst = 0.0
    for W, g in _cases():
        packed, scale, zero, shape = _quantize_pack(W, g)
        expected = _dequant_dense(packed, scale, zero, g, shape)
        try:
            got = np.asarray(
                sol.reconstruct_dense_from_affine_quantized(
                    packed.copy(), scale.copy(), zero.copy(), g, shape
                ),
                dtype=np.float64,
            )
        except Exception:
            return {"max_abs_err": float("inf")}

        if got.shape != expected.shape or not np.all(np.isfinite(got)):
            return {"max_abs_err": float("inf")}

        worst = max(worst, float(scorers.max_abs_err(expected, got)))

    return {"max_abs_err": worst}
