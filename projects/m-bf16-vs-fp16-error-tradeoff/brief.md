# Ticket: Numerical Instability and Precision Tradeoffs in Blockwise Attention

We are observing significant anomalies in our long-context LLM serving pipeline after migrating default precision tiers and scaling sequence lengths. Specifically, engineering reports indicate two distinct failure modes occurring intermittently under high-load production environments.

First, when running workloads configured with bfloat16, operators notice subtle divergence in cumulative perplexity metrics compared to fp16 baselines, particularly during deep reduction passes where dynamic range constraints interact with accumulated rounding errors. We need a robust analytic utility to measure and bound the relative error profiles between bf16 and fp16 representations across various tensor scales.

Second, during sequences involving heavy padding or specialized masking patterns, entire rows of the attention matrix are occasionally fully masked out (i.e., all entries set to negative infinity or masked). In these cases, the attention kernel is producing `NaN` values that propagate through the softmax reduction tree, crashing downstream layers or corrupting hidden states. We must ensure that fully-masked rows are handled gracefully without generating NaNs, while preserving mathematical equivalence to standard exact attention when blocks are partitioned.

Finally, we need a rigorous test suite to prove the exactness of our blockwise attention implementation against standard dense attention across different block sizes and numerical precisions, ensuring no regressions are introduced during future optimizations.
