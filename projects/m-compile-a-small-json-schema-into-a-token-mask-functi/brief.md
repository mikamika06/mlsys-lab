# Ticket: Production constrained sampling overhead and schema deadlock

## Symptom
Our edge inference microservice runs a lightweight schema-constrained generator to enforce strict structured outputs from local models. Recently, on structured calls requiring tight schemas, telemetry showed two distinct issues under peak load:

1. Throughput drops significantly when schema enforcement is enabled compared to unconstrained decoding, but we lack exact token-per-second benchmarking isolated to the masking and decode loop.
2. Certain requests using small, nested JSON Schemas cause the generation loop to freeze or hang indefinitely during decoding without emitting an end-of-sequence token or valid JSON closing brace.

## Task
Investigate the token-masking pipeline and structured generation runner in `schema_runner/`.

1. Implement a schema compiler and token-mask function (`schema_runner/compiler.py`) that constructs allowed token sets per decoding state for small JSON Schemas.
2. Build a throughput measurement utility (`schema_runner/benchmark.py`) that calculates decode tokens per second both with and without schema-constrained masking.
3. Add deadlock detection and diagnostic tools (`schema_runner/diagnostics.py`) to catch unsatisfiable schema states where no valid continuation token exists or termination is impossible.
4. Write a regression suite (`tests/test_regression.py`) that detects masked state generation failures and catches unsatisfiable/non-terminating schema configurations before deployment.
