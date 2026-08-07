# Optimal Block Size for Mixed Workloads in KV Cache Management

Our serving cluster is experiencing degraded throughput and elevated memory overhead under mixed inference workloads. Production traces show a heterogeneous mix of long-context requests that benefit from shared prefix caching and short-context ad-hoc requests. System metrics indicate that setting the KV cache block size too small inflates page table metadata and causes severe prefix cache thrashing, while setting it too large leads to massive internal fragmentation and partial-block truncation loss when matching prefix trees.

We need a systematic model to evaluate the trade-offs between internal fragmentation waste and prefix hit loss across candidate block sizes. Furthermore, we must implement a simulation sweep that models vLLM block allocation behavior under live workload distribution traces to select the optimal block size for our serving engine.

To stabilize our infrastructure, you will:
1. Build an analytical waste-and-hit-loss objective model that calculates internal fragmentation and partial-block prefix truncation loss for arbitrary sequence collections and block sizes.
2. Implement a block-size sweep simulation over request trace streams that models block allocation, prefix tree matching, and cache eviction to compute the net cost curve and identify the `argmin` block size.
3. Construct a regression test suite that validates the monotonicity of partial-block truncation, verifies fragmentation bounds, and detects regression errors when prefix hit scoring incorrectly ignores partial block tail loss.
