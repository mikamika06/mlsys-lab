import numpy as np


class AffineQuantizedTensor:
    def __init__(self, int_data: np.ndarray, scale: np.ndarray, zero_point, shape: tuple, group_size: int, asymmetric: bool):
        self.int_data = int_data
        self.scale = scale
        self.zero_point = zero_point
        self.shape = shape
        self.group_size = group_size
        self.asymmetric = asymmetric

    def dequantize(self) -> np.ndarray:
        """Dequantize to float32 numpy array of original shape."""
        raise NotImplementedError

    def __matmul__(self, other: np.ndarray) -> np.ndarray:
        """Matrix multiplication using dequantized weights."""
        raise NotImplementedError


def quantize_affine(weight: np.ndarray, group_size: int, asymmetric: bool) -> AffineQuantizedTensor:
    """
    Quantize a float32 weight matrix using group-wise affine quantization.
    Returns an AffineQuantizedTensor.
    """
    raise NotImplementedError
