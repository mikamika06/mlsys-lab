# Ticket: Causal Mask Discrepancies in FlashAttention Integration

During recent profiling runs on our FlashAttention-based causal decoding kernels, we noticed subtle correctness issues in our attention score masking logic, particularly when scaling to variable sequence lengths and handling decode steps.

Our current mask generator appears to produce incorrect attention patterns at the boundaries, causing either information leakage from future tokens or overly conservative masking that drops valid context. Specifically, when inspecting the mask matrices generated for variable query and key lengths ($S_q$ and $S_k$), we observe unexpected discrepancies between top-left and bottom-right alignment conventions, especially when dealing with single-token decode steps where $S_q = 1$.

In a standard causal mask, every query at index $i$ should only attend to keys at index $j \le i$ (under top-left indexing) or properly aligned right-shifted indices (under bottom-right alignment for block-sparse or variable-length FlashAttention kernels). When $S_q = 1$ during incremental decoding, the alignment offset frequently collapses or shifts into the wrong quadrant, leading to NaN outputs or completely masked-out attention weights depending on whether the block layout expects left-padded or right-padded sequence tensors.

We need a dedicated, exact FlashAttention causal mask generator unit that cleanly isolates and tests these behaviors across three progressive milestones:
1. Generating exact causal attention boolean/float masks for arbitrary $S_q$ and $S_k$ dimensions.
2. Computing the exact disagreement map between top-left and bottom-right causal alignment conventions.
3. Correctly handling edge cases and specialized masks for decode steps where $S_q = 1$, complete with a robust regression test suite to catch any incorrect shift or offset assumptions.

Please implement the required modules, reference solutions, test suites, and harness checkers for this exercise unit.
