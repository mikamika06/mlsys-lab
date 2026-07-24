import numpy as np

def _oracle_quantize(x):
    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    block_size = 32
    num_blocks = n // block_size
    codes = np.empty_like(x, dtype=np.int8)
    scales = np.empty(num_blocks, dtype=np.float16)
    for b in range(num_blocks):
        start = b * block_size
        end = start + block_size
        block = x[start:end]
        absmax = np.max(np.abs(block))
        d = absmax / 127.0 if absmax != 0 else 0.0
        scales[b] = np.float16(d)
        c = np.round(block / d) if d != 0 else np.zeros_like(block)
        c = np.clip(c, -127, 127).astype(np.int8)
        codes[start:end] = c
    return codes, scales

def _oracle_dequantize(codes, scales):
    block_size = 32
    num_blocks = len(scales)
    x_hat = np.empty_like(codes, dtype=np.float32)
    for b in range(num_blocks):
        start = b * block_size
        end = start + block_size
        c_block = codes[start:end].astype(np.int8).astype(np.float32)
        d = scales[b].astype(np.float32)
        x_hat[start:end] = c_block * d
    return x_hat

def grade(sol, fx) -> dict:
    ok = 1.0
    rng = np.random.default_rng(12345)
    for _ in range(5):
        n_blocks = rng.integers(2, 10)          # 2–9 blocks
        n = n_blocks * 32
        x = rng.standard_normal(n).astype(np.float32) * 10.0
        try:
            got_codes, got_scales = sol.q8_0_quantize(x)
            ref_codes, ref_scales = _oracle_quantize(x)
        except Exception:
            ok = 0.0
            break
        if not (got_codes.shape == ref_codes.shape and np.array_equal(got_codes, ref_codes)):
            ok = 0.0
            break
        if not (got_scales.shape == ref_scales.shape and np.array_equal(got_scales.astype(np.float16), ref_scales)):
            ok = 0.0
            break
        try:
            got_x_hat = sol.q8_0_dequantize(got_codes, got_scales)
            ref_x_hat = _oracle_dequantize(ref_codes, ref_scales)
        except Exception:
            ok = 0.0
            break
        if not np.array_equal(got_x_hat.astype(np.float32), ref_x_hat):
            ok = 0.0
            break
    return {"exact_match": ok}
