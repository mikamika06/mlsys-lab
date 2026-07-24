import numpy as np

FP16_MAX = 65504.0


def pick_loss_scale(grads: np.ndarray, fp16_max: float = FP16_MAX) -> float:
    """Return the largest power of two S = 2**e with max(|grads|) * S <= fp16_max.

    Return 1.0 when `grads` is all zeros.
    """
    raise NotImplementedError('your code here')


def to_fp16_grads(grads: np.ndarray, scale: float) -> np.ndarray:
    """Return grads * scale stored as a float16 array of the same shape."""
    raise NotImplementedError('your code here')


def unscale_grads(grads_fp16: np.ndarray, scale: float) -> np.ndarray:
    """Return the float32 gradients recovered from `grads_fp16` by dividing out `scale`."""
    raise NotImplementedError('your code here')
