# Symptom: High TTFT and Redundant KV Computations on Repeated Prompt Prefixes

Production serving logs for long-context requests show high Time-To-First-Token (TTFT) and severe memory thrashing even when incoming requests share identical prompt prefixes (e.g., system prompts, multi-shot examples, or long document contexts). Monitoring indicates that KV cache blocks are being allocated and recomputed from scratch on every incoming request.

Analysis of our custom engine routing layer shows that incoming tokens are not being deterministically mapped to cached block structures. Block allocation operates strictly sequentially without hashing sequence history, leading to cache misses even when exact block-aligned prefix matches exist in memory. Additionally, without a structured block hashing scheme, the engine cannot track prefix hit rates or evaluate the latency impact of automatic prefix caching.

Your task is to implement vLLM's prefix block-hashing scheme, build a prefix cache metric tracking system to measure hit rates, and write regression tests to safeguard against incorrect hash parent chain links.
