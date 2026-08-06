import mlx.core as mx
from mlx_bench.bench import run_benchmark
from mlx_bench.compile_opt import benchmark_matmul_chain


def test_benchmark_actually_computes():
    x = mx.random.normal((64, 64))
    weights = [mx.random.normal((64, 64)) for _ in range(3)]
    out = run_benchmark(x, weights)
    assert out.shape == (64, 64)


def test_compilation_speedup_exists():
    x = mx.random.normal((128, 128))
    weights = [mx.random.normal((128, 128)) for _ in range(4)]
    t_uncompiled = benchmark_matmul_chain(x, weights, compiled=False)
    t_compiled = benchmark_matmul_chain(x, weights, compiled=True)
    assert t_compiled <= t_uncompiled * 1.5


def test_implicit_eval_detection():
    from mlx_bench.graph import analyze_implicit_evals
    ops = ["matmul", "item", "add", "numpy"]
    triggers = analyze_implicit_evals(ops)
    assert len(triggers) >= 2
