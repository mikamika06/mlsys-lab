# Hardening vLLM API Endpoints Against Resource Exhaustion and Log Data Leaks

Our multi-tenant LLM serving layer experienced an incident where a single external user exhausted GPU compute budgets across shared nodes, causing severe latency spikes for concurrent tenants. An audit revealed that request validation failed to calculate worst-case execution budgets before enqueuing requests. Specifically, multi-prompt requests with large generation targets (`max_tokens`) and multiple completions (`n`) passed through without enforcing maximum bounded GPU cost constraints.

Furthermore, internal security compliance alerted us that raw prompt contents and Personally Identifiable Information (PII) are currently written into application logs under production logging configurations.

To secure the endpoint and restore predictable tenant isolation, we need to implement a three-part defense:
1. Deterministic worst-case request compute cost estimation based on prompt size, max generated tokens, and parallel output streams (`n`).
2. Admission control logic to reject requests that exceed configured maximum GPU-second bounds before allocation occurs.
3. A configurable log sanitizer that intercepts outgoing log payloads and redacts prompt text or sensitive PII patterns according to active log level settings.
