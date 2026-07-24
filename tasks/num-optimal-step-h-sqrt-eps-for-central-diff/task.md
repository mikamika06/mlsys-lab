## Context

The central-difference estimate of a derivative is

$$
D_h f(x) = \frac{f(x+h) - f(x-h)}{2h}.
$$

Its error has two competing sources:

- **Truncation error**, from the Taylor remainder: $\left|D_h f(x) - f'(x)\right| \approx \frac{h^2}{6}\,|f'''(x)|$. This *shrinks* as $h \to 0$.
- **Rounding error**, from evaluating $f$ in floating point: each evaluation carries an absolute error of order $\varepsilon\,|f(x)|$ (machine epsilon $\varepsilon \approx 2.22\times10^{-16}$ for float64), so the *quotient* $\frac{f(x+h)-f(x-h)}{2h}$ carries an error of order $\frac{\varepsilon\,|f(x)|}{h}$. This *grows* as $h \to 0$.

Total error is roughly $A h^2 + B/h$ for constants $A, B$ depending on $f$ and $x$;
balancing the two terms gives an optimal step of order $h^\star \sim \varepsilon^{1/3}$
(this is different from the familiar $\sqrt{\varepsilon}$ rule of thumb, which is
optimal for the one-sided *forward* difference $\frac{f(x+h)-f(x)}{h}$, whose
truncation error is only $O(h)$). Too small an $h$ is *not* automatically more
accurate — it can be much worse, because rounding error dominates.

## Task

Implement `best_h_central_diff(f, fprime, x, h_grid)`:

```python
def best_h_central_diff(f, fprime, x: float, h_grid: np.ndarray) -> float:
    ...
```

- `f`: a callable, `f(x) -> float`.
- `fprime`: a callable giving the true analytic derivative, `fprime(x) -> float`.
- `x`: the point at which to evaluate.
- `h_grid`: a 1-D NumPy array of candidate step sizes to search over.

For every $h$ in `h_grid`, compute the central-difference relative error

$$
\text{err}(h) = \frac{\left|D_h f(x) - f'(x)\right|}{|f'(x)|},
$$

and return the value $h^\star \in$ `h_grid` (an actual element of the array,
not an interpolated or externally chosen value) that minimizes it.

## Example

```python
import numpy as np

h_grid = np.logspace(-12, -1, 40)
h_star = best_h_central_diff(np.sin, np.cos, 1.3, h_grid)
# h_star should land somewhere around 1e-5 to 1e-6, NOT at the smallest
# h_grid value — the smallest h in the grid is dominated by rounding noise
# and gives a much larger error than the optimum.
```

## What the gate checks

The gate evaluates several `(f, fprime, x)` cases (sin/cos, exp/exp, a cubic
polynomial and its derivative, log/reciprocal at a larger `x`) against a
shared, seeded logarithmic grid of step sizes from about $10^{-12}$ to
$10^{-1}$. For each case it computes the reference optimum by brute-force
scanning the *same* grid with the formula above (a real oracle — the
analytic derivatives are exact closed forms, and the scan is exhaustive, not
hardcoded) and records the minimal achieved error `err(h_ref)`.

It then evaluates the achieved error at the `h_star` your function returns,
`err(h_your)`, and checks that `h_your` is really an element of `h_grid`.
The gate metric is the worst case over all test cases of

$$
\text{rel\_err} = \frac{\text{err}(h_{\text{your}}) - \text{err}(h_{\text{ref}})}{\text{err}(h_{\text{ref}})},
$$

i.e. how much worse your chosen step is than the true grid optimum, relative
to the optimum's own error. This must stay below $5\%$. Comparing achieved
error (rather than requiring `h_your == h_ref` bit-for-bit) tolerates
near-ties between adjacent grid points while still failing a solution that
picks a step size far from optimal — for instance, always returning the
*smallest* `h` in the grid (a common but wrong intuition) lands deep in the
rounding-error-dominated regime and misses the gate by orders of magnitude.
