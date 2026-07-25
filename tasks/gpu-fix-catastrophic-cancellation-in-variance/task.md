## Context

The "textbook" one-pass variance formula,

$$\mathrm{Var}(x) = E[x^2] - E[x]^2,$$

is exact in real-number arithmetic but numerically treacherous in
floating point whenever the mean is large relative to the true spread.
$E[x^2]$ and $E[x]^2$ are then both huge, nearly-equal numbers — and
subtracting two nearly-equal floats destroys almost all of their
significant digits (catastrophic cancellation). At the extreme, the
result can even come out **negative**, which a real variance can never
be — an unmistakable tell that the formula, not the data, is broken.
This is exactly the failure mode LayerNorm/RMSNorm kernels have to avoid:
activations routinely sit at a large, shifting mean while the quantity
that actually matters for normalization is the (much smaller) spread
around it.

The fix is to **center first**: compute the mean in one pass, then
reduce $(x_i - \text{mean})^2$ in a second pass. Every term in that sum
is already small, so there's nothing large to cancel.

## Task

`row_variance` (in `solve.cu`) reduces `x[0..n)` (one block, `n` threads)
to `out[0]` using the one-pass `E[x^2] - E[x]^2` formula — exact in real
arithmetic, catastrophic in float. Rewrite it as a two-pass reduction:
first reduce `x[]` to get the mean, `__syncthreads()`, then reduce
`(x[tid] - mean)^2` (re-reading the *original* `x[tid]`, not any
single-pass moment) to get the variance.

## Example

32 values built as `1e10 + eps` where `eps` is a handful of small
integers (`-3` to `3`): the true variance is exactly `4.8896484375`
(translation-invariant — adding `1e10` to every value can't change the
spread). The one-pass formula returns **`-16384`** on this input: a
negative variance, the clearest possible sign that `E[x^2]` and `E[x]^2`
cancelled away everything but rounding noise.

## What the gate checks

`max_abs_err <= 1e-6` against `np.var(x)` (computed by a numerically
stable two-pass algorithm) on the fixed 32-element, mean-`1e10` fixture.
The shipped one-pass kernel is off by `16388.9` — not a rounding
quibble, a sign-flipped, four-orders-of-magnitude wrong answer. Any fix
that still derives variance from two separately-accumulated moments
(rather than centering on the mean before squaring) keeps failing the
same way.
