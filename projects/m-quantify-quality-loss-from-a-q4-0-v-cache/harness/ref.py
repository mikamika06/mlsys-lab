import numpy as np

CONFIGS = [
    {"n_layers": 32, "n_kv_heads": 8, "head_dim": 128, "seq_len": 2048, "n_parallel": 1, "v_type": "q4_0"},
    {"n_layers": 32, "n_kv_heads": 8, "head_dim": 128, "seq_len": 2048, "n_parallel": 4, "v_type": "q4_0"},
    {"n_layers": 40, "n_kv_heads": 8, "head_dim": 128, "seq_len": 4096, "n_parallel": 8, "v_type": "q4_0"},
    {"n_layers": 16, "n_kv_heads": 4, "head_dim": 64, "seq_len": 1024, "n_parallel": 2, "v_type": "f16"},
]


def compute_slot_kv_bytes(n_layers, n_kv_heads, head_dim, seq_len, n_parallel, element_size_k=2, v_type="q4_0", block_size=32):
    if v_type == "f16":
        v_bytes_per_elem = 2.0
    elif v_type == "q4_0":
        v_bytes_per_elem = (2.0 + block_size // 2) / block_size
    else:
        v_bytes_per_elem = 2.0

    k_bytes_per_token = n_layers * n_kv_heads * head_dim * element_size_k
    v_bytes_per_token = int(np.ceil(n_layers * n_kv_heads * head_dim * v_bytes_per_elem))

    single_slot_bytes = (k_bytes_per_token + v_bytes_per_token) * seq_len
    total_bytes = single_slot_bytes * n_parallel
    return int(total_bytes)


def predict_multi_slot_growth(configs):
    res = []
    for cfg in configs:
        b = compute_slot_kv_bytes(
            cfg["n_layers"],
            cfg["n_kv_heads"],
            cfg["head_dim"],
            cfg["seq_len"],
            cfg["n_parallel"],
            cfg.get("element_size_k", 2),
            cfg.get("v_type", "q4_0"),
            cfg.get("block_size", 32),
        )
        res.append(b)
    return res


def quantize_q4_0(v_matrix, block_size=32):
    flat = np.asarray(v_matrix, dtype=np.float32).ravel()
    n = len(flat)
    pad = (block_size - (n % block_size)) % block_size
    if pad > 0:
        flat = np.pad(flat, (0, pad), mode="constant")

    blocks = flat.reshape(-1, block_size)
    scales = np.max(np.abs(blocks), axis=1) / 7.0
    scales[scales == 0] = 1.0

    q = np.round(blocks / scales[:, None])
    q = np.clip(q, -8, 7).astype(np.int8)

    scales_f16 = scales.astype(np.float16)

    q_unsigned = (q + 8).astype(np.uint8)
    packed = (q_unsigned[:, 1::2] << 4) | (q_unsigned[:, 0::2] & 0x0F)

    return {"scales": scales_f16, "quants": packed, "orig_shape": v_matrix.shape, "pad": pad}


def dequantize_q4_0(quantized_data, shape, block_size=32):
    scales = quantized_data["scales"].astype(np.float32)
    packed = quantized_data["quants"]

    q0 = (packed & 0x0F).astype(np.int8) - 8
    q1 = ((packed >> 4) & 0x0F).astype(np.int8) - 8

    q = np.empty((packed.shape[0], block_size), dtype=np.int8)
    q[:, 0::2] = q0
    q[:, 1::2] = q1

    dequantized = q * scales[:, None]
    flat = dequantized.ravel()

    orig_len = int(np.prod(shape))
    flat = flat[:orig_len]
    return flat.reshape(shape)


def evaluate_v_cache_loss(v_matrix, block_size=32):
    q_data = quantize_q4_0(v_matrix, block_size=block_size)
    dequant = dequantize_q4_0(q_data, v_matrix.shape, block_size=block_size)

    diff = v_matrix.astype(np.float32) - dequant.astype(np.float32)
    norm_diff = np.linalg.norm(diff)
    norm_orig = np.linalg.norm(v_matrix.astype(np.float32))

    if norm_orig == 0:
        return 0.0
    return float(norm_diff / norm_orig)
