# Stand Up 1P1D on Two GPUs and Verify Decode Prefill Skipping

We are setting up a disaggregated prefill/decode architecture across two nodes or GPU instances (1 Prefill instance, 1 Decode instance; 1P1D). In disaggregated serving, the Prefill worker executes prompt attention, constructs the KV cache, and transfers it over a connector channel to the Decode worker. When a request lands on the Decode worker, it should immediately execute token-by-token generation using the transferred KV context without re-executing full prefill compute or running redundant transformer layers over prompt tokens.

During end-to-end integration testing, we noticed that while KV transfer completes, the Decode instance continues to report heavy compute overhead and elevated prefill counts during its first iteration. Trace logs reveal that requests dispatched to the Decode worker are falling back to standard sequence processing, running prefill attention again, and ignoring the transferred KV context payload.

Your task is to fix the disaggregated routing and execution flow across the two instances:
1. Implement the 1P1D disaggregated launcher and context transfer handler in `disagg/p1d.py` so that Prefill generates KV context, transfers it to Decode, and Decode accepts it.
2. Build the execution validator in `disagg/verify.py` that verifies the Decode instance skips prefill phase steps, accurately measuring prefill step counts, transferred block counts, and relative floating point operation ratios.
3. Add regression tests in `tests/test_regression.py` that ensure any regression—such as a bypass where Decode executes full prompt prefill despite having transferred KV blocks—is caught by test failures.
