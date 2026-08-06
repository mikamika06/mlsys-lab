import numpy as np


def diagnose_and_fix_layout(
    img: np.ndarray,
    src_format: str,
    dst_format: str,
    src_order: str,
    dst_order: str,
) -> np.ndarray:
    """Fix dimension layout and color order mismatch."""
    out = img.copy()
    if src_order != dst_order:
        if src_order == "BGR" and dst_order == "RGB":
            if src_format == "NHWC":
                out = out[..., ::-1]
            elif src_format == "NCHW":
                out = out[:, ::-1, :, :]
        elif src_order == "RGB" and dst_order == "BGR":
            if src_format == "NHWC":
                out = out[..., ::-1]
            elif src_format == "NCHW":
                out = out[:, ::-1, :, :]

    if src_format != dst_format:
        if src_format == "NHWC" and dst_format == "NCHW":
            out = np.transpose(out, (0, 3, 1, 2))
        elif src_format == "NCHW" and dst_format == "NHWC":
            out = np.transpose(out, (0, 2, 3, 1))

    return np.ascontiguousarray(out)
