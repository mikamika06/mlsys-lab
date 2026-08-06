import numpy as np

TEST_WEIGHTS = [
    np.random.RandomState(0).randn(32, 128).astype(np.float32),
    np.random.RandomState(1).randn(64, 256).astype(np.float32),
    np.random.RandomState(2).randn(128, 512).astype(np.float32),
]


def quantize_int8_per_channel(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    max_vals = np.max(np.abs(w), axis=1, keepdims=True)
    scales = np.where(max_vals == 0, 1.0, max_vals / 127.0)
    q = np.clip(np.round(w / scales), -128, 127).astype(np.int8)
    return q, scales.squeeze(axis=1)


def dequantize_int8_per_channel(q: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return q.astype(np.float32) * scales[:, np.newaxis]


def quantize_int4_blockwise(w: np.ndarray, block_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    out_ch, in_ch = w.shape
    pad_len = (block_size - (in_ch % block_size)) % block_size
    if pad_len > 0:
        w_padded = np.pad(w, ((0, 0), (0, pad_len)), mode="constant")
    else:
        w_padded = w

    reshaped = w_padded.reshape(out_ch, -1, block_size)
    max_vals = np.max(np.abs(reshaped), axis=-1, keepdims=True)
    scales = np.where(max_vals == 0, 1.0, max_vals / 7.0)
    q_blocks = np.clip(np.round(reshaped / scales), -7, 7).astype(np.int8)

    q_flat = q_blocks.reshape(out_ch, -1)
    if pad_len > 0:
        q_flat = q_flat[:, :in_ch]

    return q_flat, scales.squeeze(axis=-1)


def dequantize_int4_blockwise(q: np.ndarray, scales: np.ndarray, block_size: int = 32) -> np.ndarray:
    out_ch, in_ch = q.shape
    pad_len = (block_size - (in_ch % block_size)) % block_size
    if pad_len > 0:
        q_padded = np.pad(q, ((0, 0), (0, pad_len)), mode="constant")
    else:
        q_padded = q

    num_blocks = q_padded.shape[1] // block_size
    reshaped = q_padded.reshape(out_ch, num_blocks, block_size)
    dequant = reshaped.astype(np.float32) * scales[:, :, np.newaxis]
    res = dequant.reshape(out_ch, -1)

    if pad_len > 0:
        res = res[:, :in_ch]
    return res


def compute_mse(w_orig: np.ndarray, w_dequant: np.ndarray) -> float:
    return float(np.mean((w_orig.astype(np.float32) - w_dequant.astype(np.float32)) ** 2))


def compute_bit_size(w_shape: tuple[int, int], mode: str, block_size: int = 32) -> int:
    out_ch, in_ch = w_shape
    total_elements = out_ch * in_ch

    if mode == "int8_per_channel":
        weight_bits = total_elements * 8
        scale_bits = out_ch * 32
        return weight_bits + scale_bits
    elif mode == "int4_blockwise":
        weight_bits = total_elements * 4
        num_blocks_per_row = (in_ch + block_size - 1) // block_size
        scale_bits = out_ch * num_blocks_per_row * 16
        return weight_bits + scale_bits
    else:
        raise ValueError(f"Unknown mode: {mode}")
