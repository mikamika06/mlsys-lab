# Open-AI Surface Differences and Shim Timing Recovery

We are observing subtle telemetry and operational discrepancies between native backend model serving endpoints and their OpenAI-compatible proxy shims. When routing identical streaming requests across both interfaces, downstream load balancers occasionally drop connections due to unexpected header and payload formatting divergence. Furthermore, our observability pipeline fails to recover accurate per-phase breakdown timings (such as TTFT and inter-token generation latency) when requests pass through the compatibility layer.

Additionally, our traffic router recently broke when attempting to apply advanced generation parameters that appear to be accepted by the endpoint without throwing errors, yet produce completely identical token distributions regardless of the passed parameter value.

Your task is to implement a structural diffing and audit utility in `shimdiff/diff.py`:
1. Compare raw streaming responses from a native server and an OpenAI-compatible shim, identifying structural payload field differences, extra or missing attributes, and header key variations.
2. Reconstruct per-phase execution timing metrics (Time-To-First-Token and per-token generation latencies) from raw token stream timestamp logs emitted across both endpoints.
3. Identify silent parameter ignoring by testing model responses across parameter permutations and flagging options that yield zero effective variance in output structures or logprobs.

Write a complete unit test suite in `tests/test_regression.py` that verifies structural compliance and catches invalid parameter detection logic.
