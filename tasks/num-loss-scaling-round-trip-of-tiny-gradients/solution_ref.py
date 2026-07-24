import numpy as np


def loss_scale_round_trip(grad: np.ndarray, scale: float) -> np.ndarray:
    scaled = np.asarray(grad, dtype=np.float32) * np.float32(scale)
    stored = scaled.astype(np.float16)
    restored = stored.astype(np.float32) / np.float32(scale)
    return restored.astype(np.float32)
