import numpy as np
from mpsgraph.benchmark import benchmark_mps_vs_eager
from mpsgraph.graph import MPSGraphMatMulReLU
from mpsgraph.mapping import map_recorded_sequence


def test_mpsgraph_matmul_relu_accuracy():
    a = np.ones((4, 4), dtype=np.float32)
    b = np.ones((4, 4), dtype=np.float32)
    model = MPSGraphMatMulReLU((4, 4), (4, 4))
    res = model.compare_with_numpy(a, b)
    assert res["max_abs_err"] < 1e-5


def test_op_mapping_linear_relu():
    ops = ["linear", "relu"]
    mapped = map_recorded_sequence(ops)
    assert len(mapped) == 3
    assert "matrixMultiplicationWithPrimaryTensor:secondaryTensor:name:" in mapped[0]


def test_benchmark_latencies_and_runs():
    def dummy_graph(a, b):
        return np.maximum(0, a @ b)

    def dummy_eager(a, b):
        return np.maximum(0, a @ b)

    a = np.random.randn(8, 8).astype(np.float32)
    b = np.random.randn(8, 8).astype(np.float32)
    res = benchmark_mps_vs_eager(dummy_graph, dummy_eager, (a, b), warmup=2, runs=10)
    assert res["runs"] == 10
    assert res["warmup"] == 2
    assert res["graph_latency_ms"] > 0.0
    assert res["eager_latency_ms"] > 0.0
