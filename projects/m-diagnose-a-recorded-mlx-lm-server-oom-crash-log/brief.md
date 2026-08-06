# Ticket: mlx_lm.server Crash and Quantization Drift Investigation

Our edge deployment pipeline recently experienced an unexpected crash during high-throughput inference on Apple Silicon edge devices running `mlx_lm.server`. Additionally, downstream tasks reported severe quality degradation when using low-bit quantized models converted through our automated pipeline.

## Reported Symptom
During a peak load test, `mlx_lm.server` logged an out-of-memory (OOM) fatal exception and terminated abruptly. The server was receiving concurrent requests with varying sequence lengths. We captured the server standard error log snippet leading up to the failure (`oom_crash.log`), but the exact memory threshold, sequence length trigger, and offending concurrent batch configuration have not been pinpointed.

At the same time, models processed through our quantization converter (`convert` -> `quantize` -> `dequantize`) showed erratic generation performance. We suspect that our configuration parser incorrectly reconstructs the quantization scheme (bits and group size) from model `config.json` files, or that non-linear weight degradation across quantization steps is exceeding safety tolerances.

## Goal
1. Parse and diagnose the recorded server OOM crash log to extract critical memory allocation telemetry, offending batch metrics, and failure parameters.
2. Build an inspectable model config tool that accurately reconstructs quantization bits and group size from `config.json` structures, handling missing default fallbacks.
3. Implement a weight drift simulator for round-trip `convert -> quantize -> dequantize` operations to evaluate mean squared error (MSE) weight drift against configurable safety thresholds.
4. Author a regression test suite in `tests/test_regression.py` that catches undetected weight drift or faulty bit reconstruction before edge model artifacts are deployed to production.
