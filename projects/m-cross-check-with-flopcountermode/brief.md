# Ticket: test_mfu_accounting failing in nightly CI

We have a recurring failure in our nightly CI pipeline on the `test_mfu_accounting` suite. The CI runs an empirical FLOP counting context manager (a mock `FlopCounterMode`) that simulates operations and tallies the total floating-point operations for standard attention passes. It then cross-checks this against our `analytical_flops` formula.

Currently, the tests report a significant discrepancy between the two. The `rel_err` function is returning around `0.11` (or 11% relative error) instead of exactly `0.0`. This discrepancy is bleeding into our production metrics dashboards, causing Model FLOPs Utilization (MFU) to read higher or lower than it actually is, depending on the sequence length and hidden dimension.

Users rely on accurate MFU reporting to tune batch sizes and maximize hardware utilization. The symptom is strictly numerical: the FLOP count from the `attention_forward` shape-tracker doesn't match the analytical equation.

Please implement the shape-based `FlopCounterMode` in `attention/tracker.py` and the mock operations (`matmul` and `softmax`) in `attention/ops.py`. Then, implement the analytical and empirical FLOPs in `attention/cross_check.py` and ensure they match exactly. Finally, add a regression test so we don't accidentally regress on the exact formula in the future.
