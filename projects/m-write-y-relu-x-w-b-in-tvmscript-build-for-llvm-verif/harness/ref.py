import numpy as np


class MockExportedProgram:
    def __init__(self, nodes):
        self.graph_nodes = nodes


SEED = 42


def get_test_tensors():
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal((4, 8), dtype=np.float32)
    w = rng.standard_normal((8, 6), dtype=np.float32)
    b = rng.standard_normal((6,), dtype=np.float32)
    return x, w, b


def compute_ref_relu_matmul_add(x, w, b):
    return np.maximum(np.dot(x, w) + b, 0)


EXPORTED_MODELS = [
    MockExportedProgram(["call_tir", "call_tir", "raw_op"]),
    MockExportedProgram(["raw_op", "raw_op", "raw_op", "raw_op"]),
    MockExportedProgram(["call_tir", "raw_op", "call_tir", "raw_op", "call_tir"])
]
