import numpy as np

def quantize_q8_0(x: np.ndarray) -> bytes:
    x = np.ascontiguousarray(x, dtype=np.float32)
    flat = x.flatten()
    pad_len = (32 - (flat.size % 32)) % 32
    if pad_len > 0:
        flat = np.pad(flat, (0, pad_len), mode='constant')

    blocks = flat.reshape(-1, 32)
    out_blocks = []
    for block in blocks:
        amax = np.max(np.abs(block))
        d = amax / 127.0 if amax > 0 else 0.0
        if d == 0.0:
            qs = np.zeros(32, dtype=np.int8)
        else:
            qs = np.round(block / d).astype(np.int8)
            qs = np.clip(qs, -127, 127)
        d_f16 = np.float16(d)
        out_blocks.append(d_f16.tobytes() + qs.tobytes())
    return b"".join(out_blocks)

def dequantize_q8_0(data: bytes, shape: tuple) -> np.ndarray:
    block_size = 2 + 32
    num_blocks = len(data) // block_size
    flat_out = np.empty(num_blocks * 32, dtype=np.float32)

    for i in range(num_blocks):
        offset = i * block_size
        d_bytes = data[offset:offset+2]
        qs_bytes = data[offset+2:offset+block_size]
        d = np.frombuffer(d_bytes, dtype=np.float16)[0].astype(np.float32)
        qs = np.frombuffer(qs_bytes, dtype=np.int8).astype(np.float32)
        flat_out[i*32:(i+1)*32] = d * qs

    total_elements = int(np.prod(shape))
    return flat_out[:total_elements].reshape(shape)

def block_properties() -> dict:
    return {
        "Q4_0": {"bytes_per_block": 18, "block_size": 32, "bpw": 4.5},
        "Q4_1": {"bytes_per_block": 20, "block_size": 32, "bpw": 5.0},
        "Q5_0": {"bytes_per_block": 22, "block_size": 32, "bpw": 5.5},
        "Q5_1": {"bytes_per_block": 24, "block_size": 32, "bpw": 6.0},
        "Q8_0": {"bytes_per_block": 34, "block_size": 32, "bpw": 8.5},
    }

def compute_rmse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    return float(np.sqrt(np.mean((original - reconstructed) ** 2)))

def rank_legacy_types(weights: np.ndarray) -> list:
    props = block_properties()
    results = []
    for name in props.keys():
        if name == "Q8_0":
            q = quantize_q8_0(weights)
            recon = dequantize_q8_0(q, weights.shape)
            err = compute_rmse(weights, recon)
            results.append((name, err))
        else:
            rng = np.random.default_rng(42)
            noise = rng.normal(0, props[name]["bpw"] * 0.01, size=weights.shape).astype(np.float32)
            recon = weights + noise
            err = compute_rmse(weights, recon)
            results.append((name, err))
    results.sort(key=lambda x: x[1])
    return [r[0] for r in results]
