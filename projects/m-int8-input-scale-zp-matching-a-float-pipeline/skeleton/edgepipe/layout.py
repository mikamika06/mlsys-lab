import numpy as np


def diagnose_and_fix_layout(
    img: np.ndarray,
    src_format: str,
    dst_format: str,
    src_order: str,
    dst_order: str,
) -> np.ndarray:
    """Fix dimension layout and color order mismatch."""
    raise NotImplementedError
