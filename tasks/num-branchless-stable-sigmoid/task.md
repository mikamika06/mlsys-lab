## Context

The logistic sigmoid

$$
\sigma(x) \;=\; \frac{1}{1 + e^{-x}}
$$

is mathematically well behaved on all of $\mathbb{R}$, but the textbook expression is
not. For $x = -1000$ the intermediate $e^{-x} = e^{1000}$ overflows a float64
(the largest representable double is $\approx 1.8\cdot 10^{308}$, and $e^{709}$ is
already past it). The final answer would be a perfectly ordinary $\approx 0$, yet the
computation walks through infinity to get there and raises a floating-point overflow
on the way.

The fix is the classic *exp-of-negative-magnitude* rewrite. Multiply the numerator and
denominator by $e^{x}$ when $x < 0$:

$$
\sigma(x) \;=\;
\begin{cases}
\dfrac{1}{1 + e^{-x}}, & x \ge 0,\\[2ex]
\dfrac{e^{x}}{1 + e^{x}}, & x < 0 .
\end{cases}
$$

In both branches the exponential argument is $-|x| \le 0$, so $e^{-|x|} \in (0, 1]$ and
nothing can overflow — the worst that happens is a graceful underflow to $0$.

Writing this *branchlessly* means evaluating $z = e^{-|x|}$ once for the whole array
and selecting between $1/(1+z)$ and $z/(1+z)$ with a mask, instead of taking a Python
`if` per element.

## Task

Implement `stable_sigmoid(x)`:

```python
def stable_sigmoid(x: np.ndarray) -> np.ndarray:
    ...
```

`x` is a float64 NumPy array of any shape, possibly containing values with very large
magnitude ($|x|$ up to $10^4$). Return a float64 array of the same shape holding
$\sigma(x)$, computed so that **no intermediate value ever overflows**.

The returned values must lie in $[0, 1]$ and must be exactly monotone in the sense
that $\sigma(-x) = 1 - \sigma(x)$ holds to within float64 rounding.

## Example

```python
import numpy as np

x = np.array([-1000.0, -2.0, 0.0, 2.0, 1000.0])
print(stable_sigmoid(x))
# [0.         0.11920292 0.5        0.88079708 1.        ]
```

The naive `1 / (1 + np.exp(-x))` returns the same numbers here, but it gets the first
one by computing `np.exp(1000) -> inf` and then `1/inf`, tripping an overflow warning.
Under `np.errstate(over='raise')` that version crashes; the stable one does not.

## What the gate checks

* `max_abs_err` — the output is compared elementwise against a reference computed with
  CPython's own arbitrary-precision `decimal` module at 50 significant digits (an
  oracle completely independent of float64 arithmetic). The maximum absolute error
  over ~200 seeded random points plus a batch of extreme magnitudes must be
  $< 10^{-12}$.
* `overflow_free` — the function is called once more inside
  `np.errstate(over='raise', invalid='raise', divide='raise')` on inputs including
  $\pm 10^4$. It scores `1.0` only if nothing raises.
* `range_ok` — all outputs lie in $[0, 1]$, are finite, and satisfy the reflection
  identity $\sigma(-x) + \sigma(x) = 1$ to within $10^{-12}$.
