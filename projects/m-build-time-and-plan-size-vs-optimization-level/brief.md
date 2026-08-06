# Ticket: TensorRT engine profile tradeoffs and pipeline diagnostics

Our automated TensorRT build pipeline recently started experiencing intermittent issues when deploying models to edge workers. Engineers noticed two distinct problems:

1. Certain aggressive builder optimization levels (`builder_config.builder_optimization_level`) drastically slow down engine serialization without yielding any measurable plan size or latency benefits on specific subgraphs. In some edge deployments, high optimization levels inflate plan sizes unexpectedly or cause build timeouts. We need a systematic way to analyze trade-offs between build time, resulting plan size (using `size_ratio`), and target optimization levels across candidate builder configurations.
2. When a TensorRT build fails during graph setup, network construction, builder configuration, or engine serialization, the error messages are logged as generic runtime exceptions. We need a unified failure-stage classifier that categorizes failures into exact pipeline stages (`parser`, `network`, `builder_config`, or `engine`) by inspecting stack trace signatures and error state.

Additionally, to ensure target engines remain stable across container rebuilds, we must verify engine round-trip determinism (serializing an engine to a plan and deserializing it back yields bitwise identical layer configurations and metadata summaries).

Implement the build profiling, failure diagnosis, and round-trip verification modules. Finally, write a suite of regression tests in `tests/test_regression.py` that verifies failure classification and engine round-trip behavior.
