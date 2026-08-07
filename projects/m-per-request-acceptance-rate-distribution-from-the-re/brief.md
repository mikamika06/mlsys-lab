# EAGLE-3 Acceptance Rate Distribution and Config Diagnostic

Production monitoring alerts indicated that our speculative decoding inference pipeline running EAGLE-3 showed wildly inconsistent throughput across user workloads. While aggregate benchmarks reported promising draft acceptance ratios, per-request metrics revealed severe variance in acceptance rate distributions. Additionally, recent service deployments using vLLM speculative configuration overrides encountered intermittent initialization failures and silent fallbacks to standard autoregressive execution.

To isolate the root causes and build automated safeguards, we logged raw token draft proposals, ground-truth acceptance vectors, and speculative startup configuration attempts across realistic serving workloads.

Your task is to analyze these runtime traces and build automated diagnostics:

1. Process recorded EAGLE-3 per-request logs to extract exact token-level acceptance statistics, computing cumulative accept counts, per-request mean acceptance rates, and overall workload acceptance distribution percentiles.
2. Build a diagnostic parser for vLLM `speculative_config` runtime initialization records to identify and classify 6 distinct startup outcomes (including exact model architecture mismatches, draft tensor head dimension incompatibilities, max draft length bound violations, and valid operational states).
3. Implement a comprehensive suite of regression tests in `tests/test_regression.py` that validates acceptance rate tracking bounds and detects invalid or non-monotonically non-decreasing draft acceptance invariant violations.
