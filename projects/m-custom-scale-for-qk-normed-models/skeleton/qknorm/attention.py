import numpy as np
from qknorm.config import AttentionConfig


def rms_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    raise NotImplementedError


def compute_qknorm_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    config: AttentionConfig,
) -> np.ndarray:
    raise NotImplementedError
