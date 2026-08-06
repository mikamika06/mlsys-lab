An engineering team is analyzing inference performance logs captured from Apple Silicon Macs running local LLMs via OpenVINO GenAI and onnxruntime-genai. During a recent audit of multi-threaded execution traces, performance engineers noticed severe throughput degradation and thread contention anomalies during concurrent prompt evaluations. Specifically, when multiple asynchronous requests compete for physical performance cores, the scheduling runtime occasionally assigns threads in a way that exceeds physical core bounds, triggering aggressive OS-level thread thrashing, cache thrashing, and severe latency spikes.

Your task is to build a classification and benchmarking analysis module that processes recorded run profiles, detects fair versus unfair thread-count usage across competing engines, and evaluates token generation throughput behavior on Mac hardware.

The investigation requires three key capabilities implemented across the package:
1. Parsing recorded thread-count allocations and execution logs from multiple run records to classify whether a given session's thread utilization is fair or unfair based on core concurrency limits.
2. Computing comparative performance metrics including tokens per second ratios and execution efficiency scores between OpenVINO GenAI and onnxruntime-genai across various hardware thread configurations.
3. Writing a comprehensive regression test suite that encodes core invariants about thread allocation fairness boundaries and validates that custom classification functions correctly catch invalid or malicious classification overrides.
