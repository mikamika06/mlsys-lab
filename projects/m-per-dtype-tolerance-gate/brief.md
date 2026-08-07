# Ticket: Production Gating Failure in Numerical Divergence Pipeline

## Symptom
During our latest release candidate validation for low-precision inference models, our automated deployment pipeline triggered two critical alerts. First, the per-dtype numerical gate failed on bfloat16 models, falsely flagging mathematically sound reduction operations as numerical regressions. Second, an unhandled divergence was detected between PyTorch eager execution and the `torch.compile` graph capture for a multi-head attention reduction block, but our bisection tooling failed to localize the offending node.

## Context & Details
Our continuous integration suite relies on automated gating to catch accuracy regressions before serving model weights. When evaluating low-precision accumulators (specifically `bfloat16` accumulation across $K$ sequence terms), standard FP32 relative tolerance (`rtol`) thresholds are far too strict, while static generic tolerances fail to catch real precision degradation. We need a dynamic tolerance calculator that derives `rtol` based on the theoretical roundoff error bound for length-$K$ reductions under specified dtypes (`float32`, `float16`, `bfloat16`).

Additionally, when `torch.compile` produces outputs that deviate from eager mode beyond this derived tolerance, our debugging workflow requires a deterministic bisection utility. The current tooling struggles to locate the first intermediate operation where compiled execution diverges from eager execution.

Finally, we lack a regression test suite that validates these gating utilities against broken tolerance formulas or faulty bisection logic.

## Task
1. Implement `compute_reduction_rtol` in `tolgate/tolerance.py` to dynamically compute acceptable relative tolerances for $K$-term reductions across dtypes, and `evaluate_gate` to check matrix/tensor outputs against derived limits.
2. Implement `bisect_divergence` in `tolgate/bisection.py` to locate the exact node or step index where an execution pipeline diverges beyond acceptable limits.
3. Write a robust test suite in `tests/test_regression.py` that verifies tolerance calculations, successfully detects artificial divergences, and fails when injected with faulty tolerance estimation logic.
