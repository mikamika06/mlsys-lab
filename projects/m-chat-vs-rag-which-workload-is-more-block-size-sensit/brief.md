# Benchmark Anomaly Report: Block Size Regressions Across Chat & RAG Serving

Our production KV cache engine recently updated default PagedAttention configurations. While increasing `block_size` from 16 to 32 was expected to improve memory throughput by reducing fragmentation and allocation overhead, performance metrics showed conflicting regressions across workloads.

First, performance engineers observed that under long-context RAG workloads with shared system prompts/documents, doubling `block_size` caused a noticeable drop in prefix cache hit rate. Conversely, multi-turn Chat workloads showed very little hit-rate degradation under the exact same configuration change.

Second, microbenchmarks during the decode phase revealed that while larger block sizes reduced the number of virtual block table entries, decode latency per token did not scale down as expected and occasionally spiked when sequence lengths grew large.

We need an analytical modeling package and diagnostic test suite to:
1. Measure internal fragmentation and block sensitivity differences across Chat vs RAG traffic patterns.
2. Formally explain the hit-rate drop when doubling `block_size` on non-block-aligned shared prefixes.
3. Build a block-table lookup cost model and compare estimated translation overhead against total decode step time.
4. Implement regression safeguard tests to catch unaligned prefix cache calculation bugs.
