# Wall-Clock vs Mean Accepted Length in Speculative Decoding

Our speculative decoding engine reports a promising **mean accepted length ($\tau$) of 3.4 tokens per step**, suggesting a theoretical throughput multiplier of over 3x. However, real end-to-end benchmark measurements reveal a much more modest **1.2x wall-clock speedup**.

The production engineering team needs a diagnostic tool to profile and account for this overhead gap. We suspect that raw speculative acceptance rates mask heavy GPU/CPU overheads introduced during draft generation, verification tensor operations, and context synchronization.

To fix this, you must build a profiling utility using PyTorch's profiler (`torch.profiler`) and trace analysis tools to track actual execution time across speculative decoding phases.

## Objectives
1. Implement runtime measurement functions to calculate mean accepted length alongside true wall-clock speedup and compute the unaccounted overhead ratio.
2. Build a profiler helper to trace speculative decoding steps, extracting kernel execution times for draft model inference, target model verification, and acceptance state handling.
3. Construct a trace parser to compute phase timing splits directly from recorded PyTorch profiler traces or structured trace events.
4. Add regression tests in `tests/test_regression.py` that verify your diagnostic logic correctly catches misattributed profiling phases or incorrect speedup accounting.
