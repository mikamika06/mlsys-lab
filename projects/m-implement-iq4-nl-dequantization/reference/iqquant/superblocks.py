import numpy as np

def decode_iq4_xs(superblock: bytes) -> np.ndarray:
    scale = float(superblock[0]) * 0.01
    payload = np.frombuffer(superblock[1:], dtype=np.uint8).astype(np.float32)
    return scale * (payload - 128.0)
