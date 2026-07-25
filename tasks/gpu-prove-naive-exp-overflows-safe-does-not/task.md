## Context

Softmax is $\text{softmax}(x)_d = \exp(x_d) / \sum_k \exp(x_k)$. Computed
literally, `expf` of a logit as small as `~89` already overflows a
32-bit float — and real logits (attention scores, unnormalized class
scores) can easily reach the hundreds or thousands.

The fix costs nothing mathematically: subtract the row's own maximum
logit before exponentiating.

$$
\frac{\exp(x_d)}{\sum_k \exp(x_k)} = \frac{\exp(x_d - m)}{\sum_k \exp(x_k - m)}, \qquad m = \max_k x_k
$$

for *any* constant $m$ — multiplying every term in a ratio by
$\exp(-m)$ leaves the ratio unchanged. Choosing $m$ to be the row's own
max guarantees every exponent is $\le 0$, so $\exp(x_d - m) \in (0, 1]$
— it can never overflow, no matter how large the original logits were.

## Task

Implement

```cpp
__global__ void safe_softmax_row(float* out, const float* logits, int n_rows, int D);
```

One thread per row (`i = threadIdx.x`). For row `i`:

1. Find `m = max` over `logits[i*D .. i*D+D-1]`.
2. Compute `sum = Σ expf(logits[i*D+d] - m)` for `d` in `[0, D)`.
3. Write `out[i*D+d] = expf(logits[i*D+d] - m) / sum` for every `d`.

## Example

Row `[1000.0, 1001.0, 999.0]`: naive `expf(1000.0)` already overflows a
32-bit float (`expf` saturates or errors well before that). Subtracting
the row max `m = 1001.0` first gives exponents `[-1.0, 0.0, -2.0]` —
`expf` of each is a normal, well-behaved number in `(0, 1]`, and the
resulting softmax is numerically identical to what the (unrepresentable)
naive computation would have given.

## What the gate checks

`check.py` parses `solve.cu` with the real CUDA-C frontend and runs it on
4 fixed rows of 6 logits — two of them with values around `+-1000` — and
compares the output against numpy's own (safe) softmax. Any non-finite
output, or any exception the naive computation raises trying to evaluate
`expf` of a huge logit, is treated as an outright failure, not just an
inaccurate one. It requires

$$
\mathrm{max\_abs\_err} \le 10^{-6}
$$

on ALL four rows — including the two with extreme logits. A version that
computes `expf(logit)` directly, without subtracting anything first,
fails to even produce a finite number on those two rows. Subtracting
*some* constant other than the row max (say, a fixed value like `0`, or
the row's first element) still gives the mathematically identical ratio
in principle — softmax is invariant to any constant shift — but only the
max guarantees every shifted exponent is `<= 0`; any smaller shift
constant leaves the door open to the exact same overflow the moment a
row's true max exceeds it.
