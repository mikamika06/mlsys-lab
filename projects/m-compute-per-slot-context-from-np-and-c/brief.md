# Slot Context Allocation & Metric Verification

We have received multiple complaints regarding degraded performance and unexpected out-of-memory crashes on our production `llama-server` instances. Operational metrics indicate that several deployments are suffering from severe context fragmentation and inefficient prompt-cache utilization.

Engineers have been configuring total context window (`-c`) and parallel slot counts (`-np`) without accounting for how `llama-server` calculates context allocation per slot. Consequently, incoming requests with moderate prompt lengths either trigger premature context truncation or hit physical RAM bounds under concurrent load. Furthermore, recent server configuration updates aimed at caching shared prompt prefixes do not appear to improve time-to-first-token (TTFT) metrics in downstream telemetry.

Your task is to fix the context planning logic, determine system saturation bounds, and verify telemetry tracking.

## Goals

1. Implement per-slot context calculation to derive available context per slot from total context `-c` and slot count `-np`, while respecting fixed context limits.
2. Find the `-np` saturation point given total system VRAM/RAM constraints and prompt growth characteristics, identifying the optimal parallel slot count before thrashing occurs.
3. Parse and verify prompt-cache reusability metrics from `/metrics` responses to ensure prompt-cache hits are accurately calculated and detected under load.
4. Write a regression test suite that validates prompt-cache hit ratio computations and catches mutated cache reporting logic.
