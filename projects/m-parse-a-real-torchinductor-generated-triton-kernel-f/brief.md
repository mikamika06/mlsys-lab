# Parsing and Analyzing TorchInductor Triton Kernels

Production models optimized with TorchInductor rely heavily on custom Triton kernels generated during compilation. When debugging performance regressions or auditing kernel efficiency across different compilation flags (such as standard dynamic compilation versus `--inductor.max-autotune`), machine learning systems engineers must inspect and compare generated kernel configurations.

We have observed discrepancies between kernel performance under standard TorchInductor compilation and max-autotune mode. Currently, engineers manually inspect generated Triton python scripts and timing log dumps to determine which tuning parameters were selected and why certain candidates were chosen or discarded. This manual process is error-prone, unscalable, and slows down automated kernel regression testing.

You are tasked with implementing an automated parser and diagnostic tool set in `inductor_parse/` to programmatically extract and evaluate Triton kernel configurations:

1. Parse TorchInductor-generated Triton kernel source strings to extract active tuning parameters (such as `XBLOCK`, `YBLOCK`, `RBLOCK`, `num_warps`, and `num_stages`).
2. Compare selected configurations between standard Inductor output and candidate logs produced during max-autotune runs to compute exact parameter diffs and speedup ratios.
3. Analyze raw candidate-timing logs recorded during max-autotune autotuning passes to identify the optimal (`argmin`) configuration based on benchmarked execution times.
4. Provide a robust regression test suite in `tests/test_regression.py` that validates parameter extraction invariants and correctly flags invalid or malformed kernel configurations.
