# Token Dispatch and Combine in Mixture-of-Experts Parallelism

During distributed Mixture-of-Experts (MoE) training, tokens assigned to top-k experts must be routed across ranks via all-to-all communication. Whenexpert capacity limits are enabled, surplus tokens assigned to overloaded experts are dropped, leading to inconsistent tensor shapes across processes if dispatch protocols are implemented naively.

Your task is to implement an end-to-end token dispatch and combine pipeline from scratch. You will build the dispatch/combine routing logic with capacity dropping, calculate communication volume metrics comparing MoE all-to-all against dense FFN all-reduce, and provide a regression test suite verifying capacity limits and permutation restoration invariants.

## Requirements

1. Implement `dispatch_tokens` and `combine_tokens` in `moe/dispatch.py`:
   - Map token indices to designated experts given top-k routing indices and routing weights.
   - Respect a strict per-expert `capacity` limit. If an expert receives more tokens than its capacity, drop tokens according to their arrival order (retain the first `capacity` tokens).
   - Produce global buffer index maps and routing metadata necessary to perform all-to-all transfers and subsequent combine operations.
   - `combine_tokens` must restore expert outputs back to the original token order using weight-based scatter aggregation.

2. Implement `communication_volume` in `moe/metrics.py`:
   - Compute total bytes sent across all ranks during MoE dispatch/combine vs. dense FFN all-reduce across varied sequence lengths, top-k configurations, world sizes, hidden dimensions, and expert counts.

3. Write regression tests in `tests/test_regression.py` validating that capacity dropping never exceeds maximum expert allocations and that combine operations correctly reconstruct identical inputs when routing is an identity mapping.
