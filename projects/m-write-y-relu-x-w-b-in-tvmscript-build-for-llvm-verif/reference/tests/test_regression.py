import numpy as np
from tvm_pipeline.ops import run_llvm_relu_matmul_add
from tvm_pipeline.inspect import count_relax_ops


class DummyExportedProgram:
    def __init__(self, nodes):
        self.graph_nodes = nodes


def test_relu_matmul_add_math():
    x = np.array([[1.0, -2.0], [3.0, 4.0]], dtype=np.float32)
    w = np.array([[0.5, -0.5], [1.0, 1.0]], dtype=np.float32)
    b = np.array([-0.1, 0.2], dtype=np.float32)
    res = run_llvm_relu_matmul_add(x, w, b)
    expected = np.maximum(np.dot(x, w) + b, 0)
    np.testing.assert_allclose(res, expected, atol=1e-5)


def test_inspect_op_counts():
    prog = DummyExportedProgram(["call_tir", "raw_op", "call_tir", "raw_op", "raw_op"])
    counts = count_relax_ops(prog)
    assert counts["call_tir"] == 2
    assert counts["raw_ops"] == 3
