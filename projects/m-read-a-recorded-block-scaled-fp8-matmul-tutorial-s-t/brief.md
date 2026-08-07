# Ticket: Block-scaled FP8 tutorial performance numbers look suspicious

We recently recorded a benchmark suite running a Triton-based block-scaled FP8 matrix multiplication tutorial across various shapes, comparing its TFLOPS against standard FP16 cuBLAS implementations. However, the current analysis scripts are failing to correctly extract, parse, and evaluate the performance telemetry from the log files.

When attempting to aggregate the results, engineers noticed that the reported speedups and throughput ratios fluctuate wildly or return zero, making it impossible to determine whether the block-scaled kernel actually outperforms standard cuBLAS baselines under specific configurations. Furthermore, there is no automated regression suite to ensure that future updates to the benchmark runner or log format do not silently break the analysis pipeline.

Your task is to implement the log parsing and throughput comparison module in `fp8read/parser.py` and `fp8read/analysis.py`, and provide a robust regression test in `tests/test_regression.py` that verifies the correctness of the extracted metrics and guards against structural regressions in the analysis logic.
