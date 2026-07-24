## Context

A reduction such as summing an array looks trivial, but the naive form hides a
performance trap. Accumulating into a single variable

```cpp
double s = 0.0;
for (int i = 0; i < n; i++) s += x[i];   // s_i depends on s_{i-1}
```

creates a **loop-carried dependency chain**: each `+=` cannot start until the
previous one finishes. Floating-point addition is not associative, so the
compiler is *not allowed* to reassociate this sum on its own (that would change
the rounding). The whole loop is therefore serialized at the latency of one FP
add (typically 3-4 cycles), even though a modern core can issue several adds per
cycle.

The classic fix is to keep $K$ **independent accumulators**. Each accumulator
owns a disjoint slice of the data, so their adds have no dependency between them
and the hardware can overlap them. After the loop the $K$ partials are combined:

$$
\text{total} \;=\; \sum_{j=0}^{K-1} \text{partial}[j],
\qquad
\text{partial}[j] \;=\!\!\sum_{\substack{0 \le i < n \\ i \bmod K = j}}\!\! x[i]
$$

This is the manual version of what the optimizer would do under `-ffast-math`,
but done explicitly and safely.

## Task

Implement

```cpp
double reduce_multi_acc(const double* x, int n, double* partial, int K);
```

using the fixed **strided** assignment above:

1. Accumulator $j$ (for $0 \le j < K$) receives every $K$-th element starting at
   index $j$: `partial[j] += x[i]` whenever `i % K == j`.
2. Write the $K$ partial sums into `partial[0..K)`.
3. Return the total, which equals the full sum of `x[0..n)`.

The driver builds a deterministic array of $N = 4096$ integer-valued doubles and
calls your function with $K = 4$. Because the values are integers, the result is
exact regardless of accumulation order — a correct implementation matches the
reference bit-for-bit.

## Example

For `x = {1, 2, 3, 4, 5, 6}` and `K = 3`:

```
partial[0] = x[0] + x[3] = 1 + 4 = 5
partial[1] = x[1] + x[4] = 2 + 5 = 7
partial[2] = x[2] + x[5] = 3 + 6 = 9
total      = 5 + 7 + 9      = 21
```

## What the gate checks

The driver prints the four partial sums and the grand total. The grader compiles
your `solve.cpp` with `clang++ -O2 -std=c++20`, runs it, and compares every
printed number against the reference:

$$
\mathrm{max\_abs\_err} \;=\; \max_k \lvert \hat{y}_k - y_k \rvert \;\le\; 10^{-6}
$$

Both the total **and** each partial sum must match. A single-accumulator sum
would get the total right but the partials wrong, so it fails the gate — you must
actually split the reduction across the $K$ accumulators.
