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
        if self.asymmetric:
            deq = (self.int_data.astype(np.float32) - self.zero_point) * self.scale
        else:
            deq = self.int_data.astype(np.float32) * self.scale
        return deq.reshape(self.shape)

    def __matmul__(self, other: np.ndarray) -> np.ndarray:
        return self.dequantize() @ other


def quantize_affine(weight: np.ndarray, group_size: int, asymmetric: bool) -> AffineQuantizedTensor:
    shape = weight.shape
    w = weight.reshape(-1, group_size)
    if asymmetric:
        w_min = w.min(axis=1, keepdims=True)
        w_max = w.max(axis=1, keepdims=True)
        scale = (w_max - w_min) / 15.0
        scale[scale == 0] = 1.0
        zero_point = np.round(-w_min / scale)
        int_data = np.clip(np.round(w / scale + zero_point), 0, 15).astype(np.uint8)
        return AffineQuantizedTensor(int_data, scale, zero_point, shape, group_size, asymmetric)
    else:
        w_max = np.abs(w).max(axis=1, keepdims=True)
        scale = w_max / 7.0
        scale[scale == 0] = 1.0
        int_data = np.clip(np.round(w / scale), -7, 7).astype(np.int8)
        return AffineQuantizedTensor(int_data, scale, None, shape, group_size, asymmetric)
