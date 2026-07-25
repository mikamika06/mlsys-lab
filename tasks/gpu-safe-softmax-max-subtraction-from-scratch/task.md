## Context

Softmax exponentiates every logit: $e^{x_i}$. If any logit is large — a
real possibility with unbounded attention scores or untrained weights —
$e^{x_i}$ can overflow to infinity long before the *ratio* softmax
actually cares about does. `exp(750)` alone is already larger than the
largest finite `double`; naive softmax on a row containing a `750` logit
produces `inf`, and `inf / inf` is `NaN`.

Subtracting the row's max before exponentiating fixes this without
changing the answer: softmax is shift-invariant
($\text{softmax}(x)_i = \text{softmax}(x - c)_i$ for any constant $c$),
and subtracting the max guarantees every exponent is `<= 0`, so every
`exp()` call returns a value in `(0, 1]` — never overflowing, no matter
how large the original logits were.

## Task

Implement, in real CUDA-C:

```cuda
__global__ void safe_softmax(float* out, const float* x, int n);
```

Single-threaded (`threadIdx.x == 0` only), three passes: find
`m = max(x[0..n))`; sum `s = sum(expf(x[i] - m))`; write
`out[i] = expf(x[i] - m) / s` for every `i`.

## Example

`x = [1, 2, 750]`: unsafe `exp(750)` alone already overflows to `inf`.
With max-subtraction (`m = 750`): exponents become `1-750=-749`,
`2-750=-748`, `750-750=0` — `expf(-749)` and `expf(-748)` safely
underflow to `~0`, and `expf(0) = 1`, giving `out ≈ [0, 0, 1]`: the huge
logit dominates completely, computed without a single overflow.

## What the gate checks

`max_abs_err <= 1e-6` against a numpy oracle on a fixed 10-value row
(nine modest values plus one `750.0`). Skipping the max-subtraction (or
subtracting anything other than the true max) produces `inf`/`NaN` in at
least one output, which this check explicitly detects and fails outright
regardless of the numeric tolerance.
