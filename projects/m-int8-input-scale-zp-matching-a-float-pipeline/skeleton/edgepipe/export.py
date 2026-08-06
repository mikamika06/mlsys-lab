import numpy as np


def fold_normalize_into_graph(
    weights: np.ndarray,
    bias: np.ndarray,
    mean: list[float] | np.ndarray,
    std: list[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fold channel-wise normalization parameters into first Conv2D layer weights and bias."""
    raise NotImplementedError


def process_pipeline(
    raw_img: np.ndarray,
    src_format: str,
    dst_format: str,
    src_order: str,
    dst_order: str,
    mean: list[float],
    std: list[float],
    weights: np.ndarray,
    bias: np.ndarray,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Execute integrated preprocessing and return transformed input alongside folded weights/bias."""
    raise NotImplementedError
