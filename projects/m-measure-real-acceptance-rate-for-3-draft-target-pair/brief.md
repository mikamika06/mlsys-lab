# Speculative Decoding: Draft/Target Model Pairing Analysis

We are evaluating potential draft-target model pairings for accelerating LLM inference via speculative decoding. Our initial benchmarking setup lacked a systematic metric suite for measuring actual acceptance rates, identifying tokenizer compatibility constraints, and determining throughput-optimal pairings based on latency and acceptance dynamics.

You need to construct a lightweight evaluation and selection library that:
1. Measures real token acceptance rates across draft/target pairings using speculative verification semantics.
2. Identifies and classifies pairings based on tokenizer compatibility (matching vocabulary and token mappings vs. cross-tokenizer scenarios requiring alignment).
3. Evaluates latency-acceptance trade-offs across multiple draft models paired with a target model to select the optimal draft candidate that maximizes overall generation throughput.
4. Includes a suite of regression tests to verify that your selection logic and acceptance calculations correctly enforce critical speculative decoding invariants.
