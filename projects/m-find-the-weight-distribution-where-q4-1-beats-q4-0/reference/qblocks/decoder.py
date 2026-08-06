import numpy as np


def decode_q4_block(block_bytes, qtype="Q4_0"):
    if qtype == "Q4_0":
        scale = np.frombuffer(block_bytes[0:2], dtype=np.float16)[0].astype(
            np.float32
        )
        data = block_bytes[2:]
        weights = np.zeros(32, dtype=np.float32)
        for i in range(16):
            byte = data[i]
            low = byte & 0x0F
            high = (byte >> 4) & 0x0F
            weights[i] = (low - 8) * scale
            weights[i + 16] = (high - 8) * scale
        return weights
    elif qtype == "Q4_1":
        scale = np.frombuffer(block_bytes[0:2], dtype=np.float16)[0].astype(
            np.float32
        )
        min_val = np.frombuffer(block_bytes[2:4], dtype=np.float16)[0].astype(
            np.float32
        )
        data = block_bytes[4:]
        weights = np.zeros(32, dtype=np.float32)
        for i in range(16):
            byte = data[i]
            low = byte & 0x0F
            high = (byte >> 4) & 0x0F
            weights[i] = low * scale + min_val
            weights[i + 16] = high * scale + min_val
        return weights
    raise ValueError(f"Unknown qtype {qtype}")
