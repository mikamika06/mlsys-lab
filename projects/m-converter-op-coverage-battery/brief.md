An edge runtime export pipeline is failing during deployment on a fleet of mobile accelerators. The model deployment succeeds on host CPU devices, but when exported to target edge micro-runtimes, several operators either fail during conversion or silently suffer severe latency degradation due to inefficient fallback paths.

Inspect the model operator registry and edge runtime export stack:

1. Build a converter op-coverage battery to audit an exported graph against micro-runtime operator support tables. Identify supported, decomposable, and unsupported operators across target backends.
2. Quantify the performance trade-off between custom-op fallback paths and graph rewrite/decomposition passes. Measure execution latency overheads and construct an automated cost decision policy.
3. Construct a composite-op decomposition equivalence table that rewrites complex unsupported operators into supported atomic operations while preserving mathematical output equivalence within required tolerances.

Finally, write regression tests in `tests/test_regression.py` that validate your coverage battery and verify that incorrect op-decomposition rules that break mathematical precision or graph topological constraints are properly caught.
