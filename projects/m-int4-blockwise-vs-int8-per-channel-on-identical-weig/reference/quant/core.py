import numpy as np


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
