## Context

$$\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

is invariant to shifting every logit by the same constant $c$:

$$\frac{e^{x_i - c}}{\sum_j e^{x_j - c}} = \frac{e^{x_i}e^{-c}}{\sum_j e^{x_j}e^{-c}} = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

Picking $c = \max_j x_j$ is what makes softmax safe to compute in
`float`: every exponent becomes $\le 0$, so every $e^{x_i - c} \in (0, 1]$
— no overflow, ever, no matter how large or how spread out the raw logits
are. Skip the max-subtraction and a single large logit (`exp(1000)` is
`+inf` in `float`) turns the whole output into `NaN` (`inf / inf`). This
matters even more for a vectorized kernel: a SIMD lane can't branch away
from an overflowing exponent mid-instruction, so the uniform,
branch-free max-subtraction trick is the only way to keep every lane
numerically safe at once.

## Task

Reconstruct a numerically stable softmax:

```cpp
void softmax(const float* logits, int n, float* probs);
```

1. Find $m = \max_i \text{logits}[i]$.
2. Compute $\text{probs}[i] = e^{\text{logits}[i] - m}$ for every $i$, and
   their sum $s = \sum_i \text{probs}[i]$.
3. Normalize: $\text{probs}[i] \mathrel{/}= s$.

## Example

The driver (`main.cpp`, fixed) calls `softmax` on 8 logits deliberately
spanning a huge range — three close together near the top (`1000, 999,
998.5`), a very negative outlier (`-1000`), and a large-but-not-largest
one (`500`):

```cpp
{1000.0f, 999.0f, 998.5f, -50.0f, 0.0f, 2.0f, -1000.0f, 500.0f}
```

```
p[0]=0.628532
p[1]=0.231224
p[2]=0.140244
p[3]=0.000000
...
```

The three near-top logits get essentially all the probability mass
(summing to $1.000000$); everything more than about 20 below the max
underflows cleanly to `0.0`, never to `NaN`. Skip the max-subtraction and
compute `exp(logits[i])` directly instead: `exp(1000)` is `+inf` in
`float`, so `inf / inf` poisons the first three outputs (and the `500`
one) to `nan` instead.

## What the gate checks

The grader compiles `main.cpp` + your file with `clang++ -O2 -std=c++20`,
runs it, and requires every one of the 8 printed probabilities to satisfy
`max_abs_err <= 1e-6` against the reference. Both an unstable softmax
(`NaN` outputs) and the empty starter (`0.0` for everything, summing to
`0` instead of `1`) fail immediately and by a wide margin.
