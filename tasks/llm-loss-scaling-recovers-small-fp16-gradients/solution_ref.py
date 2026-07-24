import numpy as np

FP16_MAX = 65504.0


def pick_loss_scale(grads: np.ndarray, fp16_max: float = FP16_MAX) -> float:
    """Largest power of two S = 2**e that keeps max(|grads|) * S <= fp16_max."""
    m = float(np.max(np.abs(np.asarray(grads, dtype=np.float64))))
    if m == 0.0:
        return 1.0
    best = 2.0 ** -64
    for e in range(-64, 65):
        s = 2.0 ** e
        if m * s <= fp16_max:
            best = s
    return float(best)


def to_fp16_grads(grads: np.ndarray, scale: float) -> np.ndarray:
    """Scale the fp32 gradients by `scale` and store them as float16."""
    g = np.asarray(grads, dtype=np.float32)
    return np.asarray(g * np.float32(scale), dtype=np.float16)


def unscale_grads(grads_fp16: np.ndarray, scale: float) -> np.ndarray:
    """Widen the fp16 gradients back to float32 and divide the loss scale out."""
    g = np.asarray(grads_fp16, dtype=np.float32)
    return np.asarray(g / np.float32(scale), dtype=np.float32)
