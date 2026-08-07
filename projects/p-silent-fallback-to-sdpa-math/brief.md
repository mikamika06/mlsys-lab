# Ticket: FlashAttention acceleration not kicking in during high-throughput inference

## Symptom
We recently integrated FlashAttention into our core serving pipeline to speed up attention-heavy transformer decoding and prefill workloads. However, after deploying the updated kernel dispatcher to our staging environment, profiling metrics show zero end-to-end throughput improvement. P99 latency and token generation rates remain identical to our legacy baseline.

From initial inspection of the logs, API calls return successfully without any errors, and inference output correctness matches bit-for-bit. Because the function calls complete successfully and return valid tensors, it appears at first glance that FlashAttention is actively accelerating the workload. However, empirical measurements and hardware counter analysis suggest that the optimized fast paths are not actually executing. Instead, operations seem to be quietly defaulting to a much slower execution path behind the scenes without raising any exceptions or warnings to alert the engineering team.

## Requirements
We need to diagnose why the fast attention path is being bypassed. Specifically, we must:
1. Inspect and log the actual underlying backend executed on every call.
2. Programmatically identify the exact condition or input property causing fast-path disqualification.
3. Automatically sanitize or correct input tensors (dtype, memory alignment, masking constraints) to satisfy high-performance requirements.
4. Quantify performance before and after alignment using rigorous relative ratios.
5. Audit a suite of 20 diverse configurations to guarantee zero silent fallbacks occur.
6. Enforce strict error-raising guardrails so that any unintended fallback immediately fails loudly rather than silently degrading performance.
