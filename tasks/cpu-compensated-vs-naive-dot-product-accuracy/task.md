## Context

`float` (IEEE-754 binary32) has about 7 significant decimal digits.
Every `+=` in a running sum rounds its result back down to that
precision — and if the running sum is already large, adding a much
smaller value can round away *entirely*: the small value's bits fall
below the sum's least significant representable bit, so
`sum + tiny == sum` bit-for-bit. Do that repeatedly across a long
vector where big and small terms interleave, and a naive dot product
quietly discards a big chunk of the true answer without ever raising an
error.

**Compensated summation** (Kahan, refined by Neumaier) fixes this without
switching to a wider type. It keeps a second float, `c`, that
accumulates the rounding error from each step:

$$
t = \mathrm{sum} + \mathrm{prod}, \qquad
c \mathrel{+}= \begin{cases} (\mathrm{sum} - t) + \mathrm{prod} & |\mathrm{sum}| \ge |\mathrm{prod}| \\
(\mathrm{prod} - t) + \mathrm{sum} & \text{otherwise} \end{cases}, \qquad
\mathrm{sum} = t
$$

Whichever of `sum`/`prod` is larger, `(larger - t) + smaller` recovers
exactly the bits `t` rounded away (this is what the CPU's rounding
error *is*, computed in float arithmetic itself — no wider type
involved). Adding that recovered remainder into `c` and folding
`sum + c` back in at the end recovers precision far beyond what a single
`float` accumulator can hold on its own.

## Task

Implement, both declared in `sol.hpp`:

```cpp
float naive_dot(const float* a, const float* b, int n);
float compensated_dot(const float* a, const float* b, int n);
```

`naive_dot`: plain left-to-right accumulation, `sum += a[i] * b[i]` for
each `i`, `sum` starting at `0.0f`.

`compensated_dot`: Kahan-Neumaier compensated summation of the same
products — maintain running `sum` and compensation `c` (both starting
at `0.0f`), apply the update rule above for each `i`, and return
`sum + c` after the loop.

## Example

For `n = 3`, `a = {1e8, 1, 1}`, `b = {1, 1, 1}`: the true dot product is
`1e8 + 1 + 1 = 100000002`. `naive_dot` computes `0 + 1e8 = 1e8`, then
`1e8 + 1` rounds straight back to `1e8` (float32's resolution near `1e8`
is about `8`, so `+1` vanishes) — twice — leaving `1e8` exactly, off by
`2`. `compensated_dot` captures each vanished `+1` in `c` and returns a
result within a few ULPs of the true `100000002`.

## What the gate checks

The driver builds two length-200000 float32 vectors whose products
alternate: one in every 5 is a "big" term (~$10^8$ magnitude), the rest
are "small" terms (~$10^{-8}$ magnitude) — so the big terms repeatedly
swamp the accumulator right as small ones arrive. It computes a
double-precision reference dot product directly (the harness's own
ground truth), then prints the reference, both candidates' results, and
each one's relative error against it. The grader compiles `solve.cpp`
with `clang++ -O2 -std=c++20`, runs it, and requires

$$
\mathrm{exact\_match} = 1 \iff \text{every printed number matches the reference build}
$$

On this fixture `naive_dot`'s relative error is **1.409244e-06**;
`compensated_dot`'s is **1.554744e-08** — about 90x smaller, recovered
entirely in `float` arithmetic, no `double` accumulator anywhere in
either candidate function.
