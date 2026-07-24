import numpy as np

def decompose_sdpa(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: np.ndarray | None = None,
    scale: float | None = None,
) -> np.ndarray:
    """Decompose scaled dot-product attention into ONNX-compatible primitives."""
    raise NotImplementedError("implement decompose_sdpa")
