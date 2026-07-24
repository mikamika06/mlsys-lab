import numpy as np


def _oracle(W, block_size):
    codebook = np.array(
        [-1.0, -0.6961928, -0.52507305, -0.3949175,
         -0.28444138, -0.18477343, -0.09105004, 0.0,
         0.0795803, 0.1609302, 0.2461123, 0.33791524,
         0.44070983, 0.562617, 0.72295684, 1.0],
        dtype=np.float64,
    )

    flat = np.asarray(W, dtype=np.float64).ravel()
    n = flat.size
    padded = int(np.ceil(n / block_size) * block_size)
    x = np.zeros(padded, dtype=np.float64)
    x[:n] = flat

    blocks = x.reshape(-1, block_size)
    scales = np.max(np.abs(blocks), axis=1)
    scales[scales == 0] = 1.0

    norm = blocks / scales[:, None]
    idx = np.argmin(np.abs(norm[:, :, None] - codebook[None, None, :]), axis=2)

    max_scale = np.max(scales)
    if max_scale == 0:
        scale_codes = np.zeros_like(scales, dtype=np.uint8)
        deq_scales = np.zeros_like(scales)
    else:
        scale_codes = np.rint(scales / max_scale * 255).astype(np.uint8)
        deq_scales = scale_codes.astype(np.float64) * max_scale / 255.0

    restored = (codebook[idx] * deq_scales[:, None]).reshape(-1)[:n]
    bits = 4.0 + 8.0 / block_size + 32.0 / (block_size * 256.0)
    return restored.reshape(W.shape), float(bits)


def grade(sol, fx) -> dict:
    cases = [
        (np.array([[0.5, -1.0, 2.0, 0.0]], dtype=np.float32), 2),
        (np.arange(-32, 32, dtype=np.float32).reshape(8, 8) / 7.0, 8),
        (np.array([[0.0] * 65], dtype=np.float32), 64),
        (np.random.default_rng(4).normal(size=(3, 37)).astype(np.float32), 16),
    ]

    ok = 1.0
    for W, bs in cases:
        ref_w, ref_bits = _oracle(W, bs)
        try:
            got_w, got_bits = sol.double_quant_nf4(W, bs)
        except Exception:
            ok = 0.0
            break
        if got_w.shape != W.shape:
            ok = 0.0
            break
        if not np.allclose(got_w, ref_w, rtol=1e-6, atol=1e-6):
            ok = 0.0
            break
        if not np.isclose(float(got_bits), ref_bits, rtol=0, atol=1e-12):
            ok = 0.0
            break

    return {"exact_match": ok}
