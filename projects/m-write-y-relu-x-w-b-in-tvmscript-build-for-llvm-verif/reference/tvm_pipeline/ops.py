import numpy as np


def build_relu_matmul_add_script(m: int, n: int, k: int):
    """Constructs a TVMScript string or function for y = relu(x @ w + b)."""
    code = f"""
from tvm.script import relax as R
from tvm.script import tir as T

@I.ir_module
class Module:
    @R.function
    def main(
        x: R.Tensor(({m}, {k}), dtype="float32"),
        w: R.Tensor(({k}, {n}), dtype="float32"),
        b: R.Tensor(({n},), dtype="float32")
    ) -> R.Tensor(({m}, {n}), dtype="float32"):
        cls = Module
        with R.dataflow():
            matmul = R.matmul(x, w)
            add = R.add(matmul, b)
            out = R.nn.relu(add)
            R.output(out)
        return out
"""
    return code


def run_llvm_relu_matmul_add(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Computes y = relu(x @ w + b) in pure NumPy for LLVM verification."""
    matmul = np.dot(x, w)
    add = matmul + b
    return np.maximum(add, 0)
