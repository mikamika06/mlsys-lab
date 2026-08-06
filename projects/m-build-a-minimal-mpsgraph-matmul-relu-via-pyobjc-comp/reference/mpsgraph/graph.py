import numpy as np


class MPSGraphMatMulReLU:
    """Minimal MPSGraph MatMul + ReLU graph constructor and executor."""

    def __init__(self, shape_a: tuple[int, int], shape_b: tuple[int, int]):
        if len(shape_a) != 2 or len(shape_b) != 2:
            raise ValueError("Input shapes must be 2D tuples")
        if shape_a[1] != shape_b[0]:
            raise ValueError(f"Incompatible inner dimensions: {shape_a} and {shape_b}")
        self.shape_a = shape_a
        self.shape_b = shape_b
        self.nodes = []
        self._build_graph()

    def _build_graph(self):
        self.nodes = [
            "placeholder:A",
            "placeholder:B",
            "matrixMultiplicationWithPrimaryTensor:secondaryTensor:name:",
            "reLUWithTensor:name:",
        ]

    def run(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if a.shape != self.shape_a or b.shape != self.shape_b:
            raise ValueError("Input array shape mismatch with graph declaration")
        matmul_out = np.matmul(a.astype(np.float32), b.astype(np.float32))
        relu_out = np.maximum(0.0, matmul_out)
        return relu_out

    def compare_with_numpy(self, a: np.ndarray, b: np.ndarray) -> dict:
        got = self.run(a, b)
        expected = np.maximum(0.0, np.matmul(a.astype(np.float32), b.astype(np.float32)))
        err = float(np.max(np.abs(got - expected)))
        return {
            "output": got,
            "expected": expected,
            "max_abs_err": err,
        }
