import numpy as np


def quantize_q4_0(tensor: np.ndarray) -> bytes:
    flat = tensor.astype(np.float32).flatten()
    blocks = flat.reshape(-1, 32)
    out = bytearray()
    for block in blocks:
        max_abs = np.max(np.abs(block))
        d = np.float16(max_abs / 7.0)
        out.extend(d.tobytes())
        if d == 0:
            out.extend(bytes([136] * 16))
            continue
        scaled = np.clip(np.round(block / d.astype(np.float32)), -8, 7)
        packed = (scaled + 8).astype(np.uint8)
        for i in range(16):
            val = (packed[2 * i] & 0x0F) | ((packed[2 * i + 1] & 0x0F) << 4)
            out.append(val)
    return bytes(out)


def dequantize_q4_0(data: bytes, shape: tuple) -> np.ndarray:
    size = int(np.prod(shape))
    num_blocks = size // 32
    out = np.empty(size, dtype=np.float32)
    for i in range(num_blocks):
        block_data = data[i * 18 : (i + 1) * 18]
        d = np.frombuffer(block_data[:2], dtype=np.float16)[0].astype(np.float32)
        for j in range(16):
            val = block_data[2 + j]
            out[i * 32 + 2 * j] = ((val & 0x0F) - 8) * d
            out[i * 32 + 2 * j + 1] = (((val >> 4) & 0x0F) - 8) * d
    return out.reshape(shape)


def max_abs_err(original: np.ndarray, reconstructed: np.ndarray) -> float:
    return float(np.max(np.abs(original.astype(np.float32) - reconstructed.astype(np.float32))))
