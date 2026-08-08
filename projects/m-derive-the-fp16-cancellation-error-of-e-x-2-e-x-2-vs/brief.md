Our language model's custom LayerNorm backward pass is returning NaNs during mixed-precision (float16) training. The team traced the issue to the variance calculation returning negative values right before the square root, which mathematically should be impossible since variance is strictly non-negative. We are currently using the single-pass variance formula, `E[x^2] - E[x]^2`, to save memory bandwidth by avoiding a second read of the activation tensor. We suspect catastrophic cancellation in float16 is destroying our precision, but we need concrete measurements comparing it to a two-pass `E[(x-E[x])^2]` approach to prove this is the root cause.

Simultaneously, we have a performance tracking issue. The new fused Triton softmax kernel was projected to be 3x faster than our baseline unfused softmax by eliminating intermediate memory traffic to HBM. However, our benchmark script shows it is only achieving a 1.5x speedup. We need to rigorously calculate the theoretical memory traffic (in bytes) for both the unfused and fused approaches.

Assume an unfused softmax on a 1D vector of length N does 4 memory passes (ignoring scalar reads/writes):
1. Max pass: read X
2. Exp pass: read X, write Exp
3. Sum pass: read Exp
4. Div pass: read Exp, write Y
Total memory traffic: 4N reads and 2N writes, so 6N elements. A fused softmax reads X and writes Y, so 2N elements total.

By computing the theoretical traffic and combining it with recorded execution times, we can determine the achieved memory bandwidth in GB/s. If the fused kernel is hitting the hardware bandwidth limit, the lower-than-expected speedup is simply the hardware roofline, not a kernel bug. Please implement the numerical experiments and bandwidth calculators.
