# Performance Analysis: llama-bench Quantization & Roofline Model

We are adding standardized benchmark parsing and throughput estimation to our `llama.cpp` performance regression suite. Currently, raw `llama-bench` JSON outputs are unstructured, making automated regression tracking difficult. Additionally, we lack an analytical roofline model to verify whether observed text generation (tg) throughput matches theoretical memory bandwidth bounds.

Your task is to implement a parsing and analytical framework in `llambench/` across three distinct phases:

1. **JSON Benchmark Normalization (`llambench/parser.py`)**: Parse `llama-bench` JSON output into normalized metrics, calculating separate Prompt Processing (`pp`, tokens/sec) and Text Generation (`tg`, tokens/sec) throughputs for two quantized model variants (`Q4_K_M` and `Q8_0`).
2. **Roofline Throughput Estimator (`llambench/roofline.py`)**: Compute predicted `tg` throughput from total weight memory footprint and peak hardware memory bandwidth, and evaluate the observed vs. predicted throughput ratio.
3. **Regression Safeguard (`tests/test_regression.py`)**: Write automated test cases verifying that bad benchmark metrics (such as swapped `pp` and `tg` values) are correctly flagged and caught by your test suite.

The test suite in `tests/test_regression.py` must run with `pytest` and validate system invariants.
