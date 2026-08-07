# Symptom: High Time-To-First-Token (TTFT) and Redundant Recomputation in Agentic Tool Loops

Users running multi-turn LLM agent workflows report severe performance degradation during multi-turn interactions. Profiling shows that system prompts, tool schema definitions, and early message histories are being repeatedly recomputed across consecutive turns. Even small edits or tool additions cause the serving engine to bypass existing cached prefixes and re-evaluate sequence blocks from scratch.

Existing block-level KV caching mechanisms use fixed-size block hash tables that match token blocks rigidly. When token boundaries shift slightly or when system prompts share complex tree-structured prefix relationships (such as branching agent sub-tasks), flat block hashing fails to recognize partial prefix overlaps. This results in poor KV cache hit rates, bloated memory usage, and inflated TTFT.

To address this bottleneck, you must build a Radix Tree-based KV cache manager inspired by SGLang's RadixAttention architecture. The implementation must support tree-based token sequence insertion, accurate longest-prefix matching with node splitting, LRU eviction over unpinned leaf nodes, and ref-count pinning during active generation. Finally, you will demonstrate the superiority of the Radix Cache over flat block hashing by evaluating hit-rate metrics on an agentic tool-loop trace.
