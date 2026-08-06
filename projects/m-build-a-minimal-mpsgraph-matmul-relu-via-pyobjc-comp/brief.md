We are validating our Apple Silicon edge inference pipeline that targets Metal Performance Shaders Graph (MPSGraph) via PyObjC bindings. During hardware validation, our benchmark telemetry reported severe numeric drift and erratic latency metrics when dispatching fused operator graphs compared to PyTorch MPS eager evaluation.

Analysis of our dispatch traces revealed three distinct breaking symptoms across the pipeline:

1. The low-level MatMul + ReLU graph node constructor yields incorrect output arrays when evaluated against NumPy reference calculations, indicating disconnected graph nodes or incorrect tensor handle evaluation in `mpsgraph/graph.py`.
2. The graph compiler's operation tracer raises mapping errors when lowering recorded framework ops (such as `linear`, `conv2d`, `relu`, and `layernorm`) into Apple's documented MPSGraph primitive identifiers in `mpsgraph/mapping.py`, causing graph construction to fail on valid model layers.
3. The execution benchmarker in `mpsgraph/benchmark.py` returns inaccurate speedup ratios and zeroed run counts because it fails to perform required warmup steps and timing synchronization prior to measuring execution duration.

Fix `mpsgraph/graph.py`, complete `mpsgraph/mapping.py`, correct latency profiling in `mpsgraph/benchmark.py`, and implement regression tests in `tests/test_regression.py`.
