import numpy as np
from edgepipe.layout import diagnose_and_fix_layout


def fold_normalize_into_graph(
    weights: np.ndarray,
    bias: np.ndarray,
    mean: list[float] | np.ndarray,
    std: list[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fold channel-wise normalization parameters into first Conv2D layer weights and bias."""
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)

    std_inv = 1.0 / std
    if weights.ndim == 4:
        w_folded = weights * std_inv[None, :, None, None]
    else:
        w_folded = weights * std_inv[None, :]

    b_folded = bias - np.sum(
        weights * (mean / std)[None, :, None, None], axis=(1, 2, 3)
    )

    return w_folded.astype(np.float32), b_folded.astype(np.float32)


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
    fixed_img = diagnose_and_fix_layout(
        raw_img, src_format, dst_format, src_order, dst_order
    )
    folded_w, folded_b = fold_normalize_into_graph(weights, bias, mean, std)
    return fixed_img, (folded_w, folded_b)
