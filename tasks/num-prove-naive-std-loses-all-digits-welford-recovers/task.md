## Context

The textbook "naive" one-pass variance formula

$$
\operatorname{Var}(X) = \mathbb{E}[X^2] - \bigl(\mathbb{E}[X]\bigr)^2
$$

is mathematically exact but numerically dangerous: when every $x_i$ sits far
from zero, $\mathbb{E}[X^2]$ and $(\mathbb{E}[X])^2$ are two huge, nearly-equal
floating-point numbers, and the formula subtracts them. In `float64`
arithmetic (unit roundoff $u \approx 1.1\times 10^{-16}$), representing a
value of magnitude $M$ carries an absolute rounding error of order $u\,M$. If
the true variance is $O(1)$ but $M = \mathbb{E}[X^2] = O(\mu^2)$ for a large
mean $\mu$, the subtraction's rounding error, $O(u\,\mu^2)$, can swamp the
$O(1)$ answer entirely once $\mu^2 \gtrsim 1/u$ — the naive formula then
returns something with **no correct digits at all**, not just a slightly
noisy one.

**Welford's algorithm** computes the same quantity online, one sample at a
time, by updating a running mean and a running sum of squared deviations
*from that running mean*:

$$
\delta_i = x_i - \bar x_{i-1}, \qquad
\bar x_i = \bar x_{i-1} + \frac{\delta_i}{i}, \qquad
M_{2,i} = M_{2,i-1} + \delta_i\,(x_i - \bar x_i),
$$

with $\operatorname{Var}(X) = M_{2,n}/n$. It never forms $\mathbb{E}[X^2]$, so
it never manufactures a large intermediate value to cancel — its error stays
bounded by $O(u)$ regardless of how large $\mu$ is.

## Task

Implement:

```python
def pathological_variance_input() -> np.ndarray:
    ...
```

Construct and return a 1-D NumPy array `x` (at least 8 elements, all finite)
with an **honest, non-degenerate variance** (say $\operatorname{Var}(X)
\approx 1$, not exactly $0$) chosen so that on `x`:

* the naive one-pass formula above, evaluated in `float64`, is
  **catastrophically wrong** — its relative error against the true variance
  exceeds $0.5$ (i.e. it gets less than one correct bit);
* Welford's algorithm above, evaluated in `float64` on the very same `x`,
  stays accurate to a relative error below $10^{-8}$.

You choose the mean, spread, and length of `x` — there is no fixed input
here, you are constructing the counterexample.

## Example

```python
import numpy as np

x = pathological_variance_input()
mean = np.mean(x)
naive_var = np.mean(x**2) - mean**2          # catastrophically wrong
# a properly centered computation, e.g. np.mean((x - mean)**2) or Welford,
# stays close to the true variance
```

A mean on the order of $10^{7}$–$10^{8}$ with unit spread is enough to break
`float64`'s ~16 decimal digits of precision this way; a mean near $0$ will
not — the grader will reject an `x` that fails to reproduce the pathology.

## What the gate checks

Everything is recomputed by the grader from the `x` you return — nothing
about your construction is trusted or hardcoded:

* `valid_input` — `x` is a finite 1-D array of length $\ge 8$ with a
  non-degenerate true variance (computed as the real oracle
  $\operatorname{mean}((x - \operatorname{mean}(x))^2)$, in `float64`).
* `naive_breaks` — the grader's own implementation of the naive one-pass
  formula, run on your `x`, must have relative error $> 0.5$ against that
  true variance.
* `stable_holds` — the grader's own implementation of Welford's algorithm,
  run on the same `x`, must have relative error $< 10^{-8}$ against the true
  variance.

All three must hold simultaneously for the gate to pass.
